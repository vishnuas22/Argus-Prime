"""Phase 3 — Abstention Recovery Loop (spec §4 ladder, §5.1 schema).

Inference-time flagged layer (mechanism-b, C-MECHB / memory/patterns.md:33):
on abstain/fail/contradict the court attempts bounded measured re-attempts;
a successful ladder emits a `recovered` verdict WITH a trust penalty
(C-PENALTY), a failed ladder emits an honest `needs_review` with
machine-readable reasons. Top-level p_fake_fused / abstained / trust / risk
are NEVER mutated — the original verdict is preserved in `original`.

Public entry: run(modality, original, extras) -> dict (schema argus.recovery.v1).
Trigger false (§4.1) => not_attempted with NO model calls and NO file I/O
(C-COST): the trigger predicate is pure dict arithmetic on already-loaded
data; heavy imports are lazy, inside step functions only.

Stage A shipped BOTH image-OOD signals NOT-AVAILABLE (gate_decision.json
r3_available_on=[]), so the R0 gate runs with signal="none" and reasons never
contain "image_ood"; the R0 code-path structure is kept for the day a signal
ships. Video OOD: R0 applies only to the image_court route (§4.4).
"""
import json
import math
from pathlib import Path

WS = Path(__file__).resolve().parents[2]

SCHEMA = "argus.recovery.v1"

# Penalty clamps (§5.1): P3-C1 recovered trust cap, P3-C2 recovered risk
# floor, P3-C3 needs_review trust cap, P3-C4 needs_review risk floor.
RECOVERED_TRUST_CAP = 0.6
RECOVERED_RISK_FLOOR = 0.5
NEEDS_REVIEW_TRUST_CAP = 0.2
NEEDS_REVIEW_RISK_FLOOR = 0.9

# R1 (P3-C5 max variants 2; P3-C6 q95; P3-C7 inset 0.05), first-success order.
R1_MAX_VARIANTS = 2
R1_VARIANTS = (("jpeg_q95_rescore", {"q": 95}),
               ("inset_crop_rescore", {"inset": 0.05}))
# R2 (P3-C8 max exclude 2).
R2_MAX_EXCLUDE = 2
# Audio (P3-C16 loudnorm, P3-C17 coverage floor).
LOUDNORM_AF = "loudnorm=I=-16:TP=-1.5:LRA=11"
AUDIO_COVERAGE_FLOOR = 0.25
# 2026-11-06 audit F2 fix — audio-ladder ISSUANCE GATE. The loudnorm→
# in-coverage-vote rescore is MEASURED real-ward-biased on off-manifold audio
# (controlled pairs: real 0.9396→0.0382 Δ−0.90, fake 0.9897→0.5609 Δ−0.43 —
# both pulls real-ward). A mechanism biased toward "real" may never certify
# REAL (it would be certifying its own bias); it may still certify FAKE
# (a fake-ward output runs against the pull, so it is signal, not bias).
# Issuance therefore requires BOTH: (a) rescored p fake-ward decided
# (>= 0.5 + AUDIO_ISSUE_MARGIN), and (b) |rescored − raw| <= AUDIO_DELTA_CAP
# (confirmation gate — the rescore corroborates the raw reading, never
# overrides it). Everything else keeps the honest needs_review terminal.
# Revisit after the W-AUDIO codec-aug retrain re-measures the ladder.
AUDIO_ISSUE_MARGIN = 0.15
AUDIO_DELTA_CAP = 0.15
# server.py D-035 postures (audio has no fusion-derived trust/risk).
AUDIO_POSTURE_IN_COVERAGE = (0.7, 0.0)
AUDIO_POSTURE_OOC = (0.2, 0.9)
# Video (P3-C18 frame budget, P3-C19 YuNet retry threshold).
VIDEO_FRAME_BUDGET_RETRY = 12
VIDEO_YUNET_RETRY_THRESH = 0.45

_GATE_DECISION = WS / "artifacts" / "recovery_ab" / "gate_decision.json"
_CONFLICT_META = WS / "artifacts" / "p6v10" / "stacker_meta.json"


def _trigger(modality: str, original: dict, extras: dict) -> str:
    """§4.1 trigger predicate — pure dict arithmetic, zero I/O.

    image   : abstained OR out_of_conflict
    audio   : out_of_coverage
    video   : route image_court AND (abstained OR frames_scored < ceil(frames_court/2))
              OR route video_face AND (abstained OR identity_conflict)
    """
    if modality == "image":
        if original.get("out_of_conflict"):
            return "out_of_conflict"
        if original.get("abstained"):
            return "abstained"
        return "none"
    if modality == "audio":
        # W3 wiring: short_upload outranks out_of_coverage — the padding
        # artifact makes the score meaningless regardless of coverage.
        if original.get("short_upload"):
            return "short_upload"
        return "out_of_coverage" if original.get("out_of_coverage") else "none"
    if modality == "video":
        route = original.get("route") or extras.get("route")
        if route == "image_court":
            abstained = bool(original.get("abstained"))
            frames_scored = int(original.get("frames_scored") or 0)
            frames_court = int(original.get("frames_court") or 0)
            if abstained or frames_scored < math.ceil(frames_court / 2):
                return "per_frame_abstention"
        elif route == "video_face":
            identity = bool(extras.get(
                "identity_conflict", original.get("out_of_conflict", False)))
            if identity:
                return "identity_conflict"
            if original.get("abstained"):
                return "abstained"
    return "none"


def run(modality: str, original: dict, extras: dict | None = None) -> dict:
    """Attempt the recovery ladder for a triggered verdict (§4, §5.1).

    `original` is the court verdict dict (never mutated, C-MECHB);
    `extras` carries modality-specific context (see each ladder's needs).
    """
    extras = extras or {}
    trigger = _trigger(modality, original, extras)
    if trigger == "none":
        return _result(trigger, "not_attempted", [], [], original, None,
                       _ood_none(), original.get("trust"), original.get("risk"))
    if modality == "image":
        return _ladder_image(trigger, original, extras)
    if modality == "audio":
        return _ladder_audio(trigger, original, extras)
    if modality == "video":
        return _ladder_video(trigger, original, extras)
    return _result(trigger, "not_attempted", [], [], original, None,
                   _ood_none(), original.get("trust"), original.get("risk"))


# ---------------- assembly helpers (stdlib only) ----------------

def _ood_none() -> dict:
    return {"signal": "none", "fired": False, "value": None, "threshold": None}


def _snapshot(original: dict) -> dict:
    return {"p_fake_fused": original.get("p_fake_fused"),
            "abstained": original.get("abstained"),
            "out_of_conflict": original.get("out_of_conflict"),
            "trust": original.get("trust"),
            "risk": original.get("risk")}


def _clamp_recovered(trust, risk):
    return (None if trust is None else min(float(trust), RECOVERED_TRUST_CAP),
            None if risk is None else max(float(risk), RECOVERED_RISK_FLOOR))


def _clamp_review(trust, risk):
    return (None if trust is None else min(float(trust), NEEDS_REVIEW_TRUST_CAP),
            None if risk is None else max(float(risk), NEEDS_REVIEW_RISK_FLOOR))


def _result(trigger, outcome, steps, reasons, original, verdict, ood, trust, risk):
    return {"schema": SCHEMA, "trigger": trigger, "outcome": outcome,
            "attempted_steps": steps, "reasons": reasons,
            "original": _snapshot(original), "verdict": verdict,
            "ood": ood, "trust": trust, "risk": risk}


def _review(trigger, steps, reasons, original, ood=None, base=None,
            verdict_extra=None) -> dict:
    if not reasons:
        reasons.append("recovery_exhausted")
    bt, br = base if base is not None else (original.get("trust"),
                                            original.get("risk"))
    t, r = _clamp_review(bt, br)
    verdict = {"p_fake_fused": original.get("p_fake_fused"), "abstained": True,
               "trust": t, "risk": r, "coverage_fraction": None,
               "per_identity": None}
    if verdict_extra:
        verdict.update({k: v for k, v in verdict_extra.items()
                        if v is not None})
    return _result(trigger, "needs_review", steps, reasons, original, verdict,
                   ood or _ood_none(), t, r)


def _recovered(trigger, steps, reasons, original, vdict, ood=None) -> dict:
    t, r = _clamp_recovered(vdict.get("trust"), vdict.get("risk"))
    verdict = {"p_fake_fused": vdict.get("p_fake_fused"),
               "abstained": bool(vdict.get("abstained", False)),
               "trust": t, "risk": r,
               "coverage_fraction": vdict.get("coverage_fraction"),
               "per_identity": vdict.get("per_identity")}
    return _result(trigger, "recovered", steps, reasons, original, verdict,
                   ood or _ood_none(), t, r)


# ---------------- image ladder (§4.2) ----------------

def _conflict_p99() -> float:
    """P3-C9: conflict threshold read from stacker_meta.json at runtime."""
    try:
        return float(json.loads(_CONFLICT_META.read_text())["conflict_p99"])
    except Exception:
        return 0.163132


def _r0_ood_gate(steps: list) -> dict:
    """R0 image-OOD gate. Stage A shipped both signals NOT-AVAILABLE, so the
    gate runs with signal="none" and never fires (§4.2 keeps the structure)."""
    ood = _ood_none()
    try:
        note = json.loads(_GATE_DECISION.read_text()).get("r3_note")
    except Exception:
        note = None
    steps.append({"step": "R0", "name": "ood_gate", "ok": True,
                  "detail": {"signal": "none",
                             "note": note or "no image-OOD signal shipped "
                                             "(gate_decision.json)"}})
    return ood


def _r1_rescore(original, extras, steps) -> dict | None:
    """R1 alternate-preprocessing re-score (P3-C5/C6/C7): V1 JPEG q95 via
    run_benchmark.encode, V2 center-crop inset 5%; first success in order."""
    pil = extras.get("pil")
    modules = extras.get("modules")
    estimator = extras.get("estimator")
    stacker_clfs = extras.get("stacker_clfs")
    if pil is None or modules is None or estimator is None or stacker_clfs is None:
        steps.append({"step": "R1", "name": "alternate_preprocessing_rescore",
                      "ok": False, "detail": {"error": "missing_extras"}})
        return None
    import shutil
    import tempfile
    from run_benchmark import run_image
    art = Path(tempfile.mkdtemp(prefix="rec_r1_art_"))
    tmp = Path(tempfile.mkdtemp(prefix="rec_r1_src_"))
    try:
        for i, (name, spec) in enumerate(R1_VARIANTS[:R1_MAX_VARIANTS]):
            try:
                row, _ = run_image(modules, estimator, pil, spec, art,
                                   f"recovery_r1_{i}", tmp, stacker_clfs,
                                   original_raw=extras.get("original_raw"),
                                   return_extras=True)
            except Exception as e:
                steps.append({"step": "R1", "name": name, "ok": False,
                              "detail": {"error": str(e)[:160]}})
                continue
            ok = (not row.get("abstained")) and (not row.get("out_of_conflict", False))
            pf = None if row.get("p_real") is None else round(1.0 - row["p_real"], 4)
            steps.append({"step": "R1", "name": name, "ok": bool(ok),
                          "detail": {"p_fake_fused": pf,
                                     "abstained": bool(row.get("abstained")),
                                     "out_of_conflict": bool(row.get("out_of_conflict", False)),
                                     "trust": row.get("trust"),
                                     "risk": row.get("risk")}})
            if ok:
                return {"p_fake_fused": pf, "abstained": False,
                        "trust": row.get("trust"), "risk": row.get("risk"),
                        "coverage_fraction": None, "per_identity": None}
        return None
    finally:
        shutil.rmtree(art, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)


def _r2_refuse(original, extras, steps) -> dict | None:
    """R2 contradiction-aware re-fuse (§4.2): drop top-≤2 participation-mass
    modules, recompute k', WeightedVotingJudge + conformal_set + trust_score +
    risk_score (pure imports from vendored fusion.py). Success = len(cset)==1
    AND k'≤conflict_p99; nothing original mutated."""
    outputs = extras.get("outputs")
    d = extras.get("d")
    survivors = extras.get("survivors")
    if outputs is None or d is None or survivors is None:
        steps.append({"step": "R2", "name": "contradiction_aware_refuse",
                      "ok": False, "detail": {"error": "missing_extras"}})
        return None
    from fusion import (WeightedVotingJudge, build_contradictions,
                        conflict_metric, conformal_set, risk_score, trust_score)
    p99 = _conflict_p99()
    mass: dict = {}
    for c in build_contradictions(survivors):
        for mid in c.modules:
            mass[mid] = mass.get(mid, 0.0) + c.conflict_contribution
    ranked = sorted(mass.items(), key=lambda kv: -kv[1])
    remaining = list(survivors)
    excluded: list = []
    while True:
        k_now = conflict_metric(remaining)
        if (k_now <= p99 or len(excluded) >= R2_MAX_EXCLUDE
                or len(remaining) < 3):
            break
        drop = next((mid for mid, _ in ranked if mid not in excluded
                     and any(m.module_id == mid for m in remaining)), None)
        if drop is None:
            break
        excluded.append(drop)
        remaining = [m for m in remaining if m.module_id != drop]
    k_prime = conflict_metric(remaining)
    probs = WeightedVotingJudge().decide(remaining)
    cset = conformal_set(probs)
    t = trust_score(d, remaining, len(outputs), len(cset), k_prime)
    r = risk_score(d, outputs, build_contradictions(remaining))
    ok = (len(cset) == 1 and k_prime <= p99)
    pf = round(1.0 - probs["camera_original"], 4)
    steps.append({"step": "R2", "name": "contradiction_aware_refuse",
                  "ok": bool(ok),
                  "detail": {"excluded_modules": excluded, "k_prime": k_prime,
                             "conflict_p99": p99, "hypothesis_set": list(cset),
                             "p_fake_fused": pf, "trust": t, "risk": r}})
    if ok:
        return {"p_fake_fused": pf, "abstained": False, "trust": t, "risk": r,
                "coverage_fraction": None, "per_identity": None}
    return None


def _ladder_image(trigger, original, extras) -> dict:
    steps, reasons = [], []
    ood = _r0_ood_gate(steps)
    if ood.get("fired"):                      # would skip to R4 (never today)
        reasons.append("image_ood")
        return _review(trigger, steps, reasons, original, ood)
    v = _r1_rescore(original, extras, steps)
    if v is not None:
        return _recovered(trigger, steps, reasons, original, v, ood)
    if original.get("out_of_conflict"):
        v = _r2_refuse(original, extras, steps)
        if v is not None:
            return _recovered(trigger, steps, reasons, original, v, ood)
        reasons.append("r2_conflict_unresolved")
    reasons.append("r1_variants_failed")
    return _review(trigger, steps, reasons, original, ood)


# ---------------- audio ladder (§4.3) ----------------

def _lazy_scripts():
    import sys
    for p in (str(WS / "deepfake"), str(WS / "deepfake" / "bench"),
              str(WS / "deepfake" / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _a1_loudnorm(path, steps, cleanup) -> dict | None:
    """A1 loudnorm+16k-mono cleanup re-score (P3-C16); reuses the process-
    cached XLS-R (detect_audio's _get_audio_model). Success = cleaned
    out_of_coverage==False."""
    if path is None:
        steps.append({"step": "A1", "name": "loudnorm_rescore", "ok": False,
                      "detail": {"error": "missing_extras.path"}})
        return None
    import subprocess
    import tempfile
    _lazy_scripts()
    from video_score import _ffmpeg_bin
    ff = _ffmpeg_bin()
    if ff is None:
        steps.append({"step": "A1", "name": "loudnorm_rescore", "ok": False,
                      "detail": {"error": "ffmpeg_unavailable"}})
        return None
    tdir = Path(tempfile.mkdtemp(prefix="rec_a1_"))
    cleanup.append(tdir)
    out = tdir / "clean.wav"
    cmd = [str(ff), "-y", "-i", str(path), "-af",
           f"{LOUDNORM_AF},aresample=16000", "-ac", "1", "-ar", "16000",
           str(out)]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception as e:
        steps.append({"step": "A1", "name": "loudnorm_rescore", "ok": False,
                      "detail": {"error": str(e)[:160]}})
        return None
    if p.returncode != 0 or not out.exists():
        steps.append({"step": "A1", "name": "loudnorm_rescore", "ok": False,
                      "detail": {"error": "ffmpeg_failed",
                                 "stderr": p.stderr.decode(errors="replace")[-200:]}})
        return None
    cleaned = _rescore_audio(out)
    ok = cleaned is not None and not cleaned.get("out_of_coverage", False)
    steps.append({"step": "A1", "name": "loudnorm_rescore", "ok": bool(ok),
                  "detail": {"out_of_coverage": (cleaned or {}).get("out_of_coverage"),
                             "znorm_mean": (cleaned or {}).get("znorm_mean")}})
    return cleaned


def _rescore_audio(path) -> dict | None:
    _lazy_scripts()
    from argus_detect import detect_audio
    return detect_audio(Path(path), recovery_enabled=False)


def _a2_coverage_vote(scored: dict, steps) -> dict:
    """A2 segment-level in-coverage vote: coverage_fraction = mean(z≤thr);
    success iff ≥0.25 (P3-C17); p = mean of in-coverage segment probs only."""
    zlist = scored.get("per_segment_znorm") or []
    probs = scored.get("per_segment") or []
    thr = scored.get("coverage_threshold")
    if not zlist or thr is None or not probs:
        steps.append({"step": "A2", "name": "in_coverage_segment_vote",
                      "ok": False,
                      "detail": {"error": "missing_znorm_or_threshold"}})
        return {"p_fake_fused": None, "coverage_fraction": None}
    in_cov = [p for z, p in zip(zlist, probs) if z <= thr]
    cov = round(len(in_cov) / len(zlist), 4)
    pf = round(sum(in_cov) / len(in_cov), 4) if in_cov else None
    ok = cov >= AUDIO_COVERAGE_FLOOR
    steps.append({"step": "A2", "name": "in_coverage_segment_vote", "ok": bool(ok),
                  "detail": {"coverage_fraction": cov, "coverage_threshold": thr,
                             "n_segments": len(zlist),
                             "n_in_coverage": len(in_cov), "p_fake_fused": pf}})
    return {"p_fake_fused": pf, "coverage_fraction": cov}


def _audio_issuable(rescored_p: float, raw_p) -> tuple:
    """2026-11-06 audit F2: issuance gate for the audio recovery ladder (see
    the constant block above for the measured-bias rationale). Returns
    (ok, refusal_reason)."""
    if rescored_p < 0.5 + AUDIO_ISSUE_MARGIN:
        return False, "rescore_real_ward_bias_guard"
    # 1e-9 keeps the documented "|Δ| <= 0.15" semantics exact at the boundary
    # (raw float math: 0.65 - 0.50 = 0.15000000000000002 > 0.15 would refuse).
    if raw_p is not None and abs(rescored_p - float(raw_p)) > AUDIO_DELTA_CAP + 1e-9:
        return False, "rescore_delta_exceeds_cap"
    return True, None


def _ladder_audio(trigger, original, extras) -> dict:
    steps, reasons = [], []
    if trigger == "short_upload":
        # W3 wiring (ruled W2-(ii)): sub-2s uploads sit in the padding regime
        # (W2 padprobe 2026-10-19: 1.0s trims median z 75.0, 75% OOC, 25% vote
        # p>=0.5 on a padding artifact). No ladder step can add duration — the
        # honest terminal is needs_review at the OOC posture.
        reasons.append("short_upload_below_2s")
        return _review(trigger, steps, reasons, original, base=AUDIO_POSTURE_OOC,
                       verdict_extra={"duration_s": original.get("duration_s")})
    path = extras.get("path")
    cleanup: list = []
    try:
        cleaned = _a1_loudnorm(path, steps, cleanup)
        a2_src = cleaned if cleaned is not None else extras.get("audio_result")
        if a2_src is None:
            reasons.append("audio_ooc_unrecovered")
            return _review(trigger, steps, reasons, original,
                           base=AUDIO_POSTURE_OOC)
        a2 = _a2_coverage_vote(a2_src, steps)
        cov = a2.get("coverage_fraction")
        pf = a2.get("p_fake_fused")
        if cov is not None and cov >= AUDIO_COVERAGE_FLOOR and pf is not None:
            ok, why = _audio_issuable(pf, original.get("p_fake_fused"))
            if ok:
                t, r = _clamp_recovered(*AUDIO_POSTURE_IN_COVERAGE)
                return _result(trigger, "recovered", steps, reasons, original,
                               {"p_fake_fused": pf, "abstained": False,
                                "trust": t, "risk": r, "coverage_fraction": cov,
                                "per_identity": None},
                               _ood_none(), t, r)
            # 2026-11-06 F2: the rescore landed but is not certifiable (bias
            # guard or delta cap) — the A2 step detail above already records
            # the rescored p for the receipt. Honest terminal: needs_review.
            reasons.append(why)
        reasons.append("audio_ooc_unrecovered")
        return _review(trigger, steps, reasons, original, base=AUDIO_POSTURE_OOC,
                       verdict_extra={"coverage_fraction": cov})
    finally:
        import shutil
        for d in cleanup:
            shutil.rmtree(d, ignore_errors=True)


# ---------------- video ladder (§4.4) ----------------

def _reenter_video(path, n_frames, threshold=None) -> dict | None:
    """V1/V2 re-enter through the same public detect_video — no second court."""
    if path is None:
        return None
    _lazy_scripts()
    from argus_detect import detect_video
    if threshold is None:
        return detect_video(Path(path), n_frames, recovery_enabled=False)
    from video_face_track import set_score_threshold
    old = set_score_threshold(threshold)          # P3-C19
    try:
        return detect_video(Path(path), n_frames, recovery_enabled=False)
    finally:
        set_score_threshold(old)                  # restore 0.55 (C-HARNESS-safe)


def _v2_retrack(path, n_frames, steps) -> dict | None:
    res = _reenter_video(path, n_frames, threshold=VIDEO_YUNET_RETRY_THRESH)
    ok = res is not None and (res.get("route") == "video_face"
                              or not res.get("abstained"))
    steps.append({"step": "V2", "name": "yunet_retrack_045", "ok": bool(ok),
                  "detail": {"route": (res or {}).get("route"),
                             "abstained": (res or {}).get("abstained"),
                             "p_fake_fused": (res or {}).get("p_fake_fused"),
                             "trust": (res or {}).get("trust"),
                             "risk": (res or {}).get("risk")}})
    return res


def _per_identity(extras) -> list:
    vis = extras.get("visual") or {}
    rows = []
    for r in (vis.get("identities") or []):
        rows.append({"id": r.get("id"), "frames": r.get("frames"),
                     "mean": r.get("mean"), "topk_mean": r.get("topk_mean"),
                     "obs_only": r.get("obs_only")})
    return rows


def _ladder_video(trigger, original, extras) -> dict:
    steps, reasons = [], []
    route = original.get("route") or extras.get("route")
    path = extras.get("path")
    n_first = int(extras.get("n_frames") or 0)
    if route == "video_face":
        if trigger == "identity_conflict":
            # V3 terminal: identity_conflict ALWAYS ends needs_review (§4.4)
            steps.append({"step": "V1", "name": "frame_budget_retry_12",
                          "ok": False,
                          "detail": {"skipped": "not_image_court_route"}})
            steps.append({"step": "V2", "name": "yunet_retrack_045", "ok": False,
                          "detail": {"skipped": "route_already_video_face"}})
            steps.append({"step": "V3", "name": "identity_split", "ok": False,
                          "detail": {"n_identities": original.get("n_identities"),
                                     "n_voting_identities":
                                         original.get("n_voting_identities")}})
            reasons.append("v3_identity_conflict")
            return _review(trigger, steps, reasons, original,
                           verdict_extra={"per_identity": _per_identity(extras)})
        steps.append({"step": "V1", "name": "frame_budget_retry_12", "ok": False,
                      "detail": {"skipped": "not_image_court_route"}})
        res = _v2_retrack(path, n_first or VIDEO_FRAME_BUDGET_RETRY, steps)
        if res is not None and not res.get("abstained"):
            return _recovered(trigger, steps, reasons, original,
                              {"p_fake_fused": res.get("p_fake_fused"),
                               "abstained": False, "trust": res.get("trust"),
                               "risk": res.get("risk")})
        reasons.append("v2_retrack_failed")
        return _review(trigger, steps, reasons, original)

    abstained = bool(original.get("abstained"))
    frames_scored = int(original.get("frames_scored") or 0)
    frames_court = int(original.get("frames_court") or 0)
    if n_first >= VIDEO_FRAME_BUDGET_RETRY:       # P3-C18: skipped if >=12
        steps.append({"step": "V1", "name": "frame_budget_retry_12", "ok": False,
                      "detail": {"skipped": f"first_pass_budget_{n_first}"}})
    else:
        res = _reenter_video(path, VIDEO_FRAME_BUDGET_RETRY)
        ok = (res is not None and not res.get("abstained")
              and int(res.get("frames_scored") or 0) >= 2)
        steps.append({"step": "V1", "name": "frame_budget_retry_12", "ok": bool(ok),
                      "detail": {"frames_scored": (res or {}).get("frames_scored"),
                                 "abstained": (res or {}).get("abstained")}})
        if ok:
            # image_court caps (trust<=0.4/risk>=0.6) are kept + penalty clamp
            return _recovered(trigger, steps, reasons, original,
                              {"p_fake_fused": res.get("p_fake_fused"),
                               "abstained": False, "trust": res.get("trust"),
                               "risk": res.get("risk")})
    res = _v2_retrack(path, n_first or VIDEO_FRAME_BUDGET_RETRY, steps)
    if res is not None and (res.get("route") == "video_face"
                            or not res.get("abstained")):
        return _recovered(trigger, steps, reasons, original,
                          {"p_fake_fused": res.get("p_fake_fused"),
                           "abstained": bool(res.get("abstained")),
                           "trust": res.get("trust"), "risk": res.get("risk")})
    # V4 no-face fallback legibility (never loosen caps)
    legible = {"route": original.get("route"),
               "confidence_degraded": original.get("confidence_degraded"),
               "frames_court": frames_court, "frames_scored": frames_scored}
    still_abstained = abstained or (res is not None and res.get("abstained"))
    steps.append({"step": "V4", "name": "no_face_fallback_legibility",
                  "ok": not still_abstained, "detail": legible})
    if still_abstained:
        reasons.append("no_face_route_abstained")
    reasons.append("v1_budget_retry_failed" if n_first < VIDEO_FRAME_BUDGET_RETRY
                   else "v1_budget_skipped")
    return _review(trigger, steps, reasons, original, verdict_extra=legible)
