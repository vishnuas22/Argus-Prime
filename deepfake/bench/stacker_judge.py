"""StackerJudge - pluggable fusion judge backed by the trained P6 stacker.

Mirrors WeightedVotingJudge's interface (decide(survivors) -> hypothesis probs).
Constructed per-image with the triage capacity so the 3-dim capacity block of
the training features is reproduced exactly. module_ids MUST match the
stacker's training feature space (stored in stacker_meta.json) — the caller
loads it from there; there is no default module space.
"""
from pathlib import Path

import numpy as np

CAPS = ["HIGH", "MODERATE", "LOW"]


class StackerJudge:
    name = "histgbm-stack-v1"

    def __init__(self, clf, capacity: str, module_ids: list | None = None,
                 feature_names: list | None = None, conflict_p99: float | None = None):
        if not module_ids:
            raise ValueError(
                "StackerJudge requires module_ids (from stacker_meta.json) — "
                "no default module space; pass the stacker's module_ids explicitly")
        self.clf = clf
        self.capacity = capacity
        self.module_ids = list(module_ids)
        if feature_names is None:
            # legacy 25-name space: interleaved e/r per module, then the cap block
            feature_names = [n for m in self.module_ids for n in (f"{m}_e", f"{m}_r")] + \
                [f"cap_{c}" for c in CAPS]
        self.feature_names = list(feature_names)
        self.conflict_p99 = conflict_p99
        self.out_of_conflict = False

    def decide(self, survivors):
        self.out_of_conflict = False
        from fusion import build_contradictions, conflict_metric
        from schemas import HYPOTHESES

        by_id = {m.module_id: m for m in survivors}
        values = {}
        for mid in self.module_ids:
            m = by_id.get(mid)
            if m is None or getattr(m, "unavailable_reason", None):
                values[f"{mid}_e"] = 0.0
                values[f"{mid}_r"] = 0.0
            else:
                values[f"{mid}_e"] = float(m.evidence_score)
                values[f"{mid}_r"] = float(m.reliability_score)
        for c in CAPS:
            values[f"cap_{c}"] = 1.0 if c == self.capacity else 0.0
        values["n_survivors"] = float(len(survivors))
        contradictions = build_contradictions(survivors)
        values["contradiction_count"] = float(len(contradictions))
        values["contradiction_mass"] = round(
            sum(c.conflict_contribution for c in contradictions), 4)
        vec = []
        for name in self.feature_names:
            if name not in values:
                raise ValueError(f"feature {name!r} not computable")
            vec.append(values[name])

        # conflict gate (mechanism b): train-p99 advisory flag, NON-destructive.
        # A high conflict metric marks the verdict out_of_conflict (surfaced
        # downstream) but does NOT overwrite the stacker's probabilities —
        # forcing uniform cost ~-0.006 flux AUC and the D-035 abstention is
        # honored via the advisory flag + existing reliability-weighted fuse.
        k = conflict_metric(survivors)
        self.out_of_conflict = bool(
            self.conflict_p99 is not None and k > float(self.conflict_p99))

        x = np.array(vec).reshape(1, -1)
        p_fake = float(self.clf.predict_proba(x)[0, 1])
        p_syn = float(np.clip(p_fake, 0.001, 0.999))
        lam = 0.0
        for m in survivors:
            if m.verdict_direction == "manipulated" and any(a.location for a in m.artifacts):
                lam = max(lam, abs(m.evidence_score) * m.reliability_score)
        p_manip = p_syn * min(0.85, lam * 1.2)
        return {
            "camera_original": round(1.0 - p_syn, 4),
            "ai_generated": round(p_syn - p_manip, 4),
            "manipulated": round(p_manip, 4),
        }
