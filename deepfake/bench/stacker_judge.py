"""StackerJudge - pluggable fusion judge backed by the trained P6 stacker.

Mirrors WeightedVotingJudge's interface (decide(survivors) -> hypothesis probs).
Constructed per-image with the triage capacity so the 3-dim capacity block of
the training features is reproduced exactly.
"""
from pathlib import Path

import numpy as np

MODULE_IDS = ["metadata", "compression_history", "spectral_probe",
              "real_distribution_probe", "perturbation_probe",
              "learned_aigc_probe", "learned_face_probe"]
CAPS = ["HIGH", "MODERATE", "LOW"]


class StackerJudge:
    name = "histgbm-stack-v1"

    def __init__(self, clf, capacity: str):
        self.clf = clf
        self.capacity = capacity

    def decide(self, survivors):
        by_id = {m.module_id: m for m in survivors}
        vec = []
        for mid in MODULE_IDS:
            m = by_id.get(mid)
            if m is None or getattr(m, "unavailable_reason", None):
                vec += [0.0, 0.0]
            else:
                vec += [float(m.evidence_score), float(m.reliability_score)]
        cap_vec = [1.0 if c == self.capacity else 0.0 for c in CAPS]
        x = np.array(vec + cap_vec).reshape(1, -1)
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
