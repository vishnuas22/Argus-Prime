"""Evidence-channel registry — single source of truth for optional probes.

The court always runs the 5 core forensic modules (Argus_Master registry:
metadata, compression_history, spectral_probe, real_distribution_probe,
perturbation_probe). Optional LEARNED channels are registered here; every
consumer (benchmark harness, stacker trainer, stacker judge, detection CLI)
derives its module set from this registry so feature spaces cannot
silently diverge (D-021/D-024).

Adding a channel:
  1. implement an EvidenceModule (see learned_aigc_probe.py)
  2. register it in OPTIONAL_CHANNELS below
  3. run: .venv/bin/python deepfake/scripts/restack.py
     (re-dump -> stacker retrain -> regression gates -> promote/hold)
"""
from pathlib import Path

WS = Path(__file__).resolve().parents[2]

CORE_IDS = ["metadata", "compression_history", "spectral_probe",
            "real_distribution_probe", "perturbation_probe"]


def _face_factory(joblib_path, module_id, provenance=None):
    from learned_face_probe import LearnedFaceProbe
    kw = {"module_id": module_id}
    if provenance:
        kw["provenance"] = provenance
    return LearnedFaceProbe(joblib_path, **kw)


def _aigc_factory(joblib_path, module_id=None, provenance=None):
    from learned_aigc_probe import LearnedAIGCProbe
    kw = {}
    if module_id:
        kw["module_id"] = module_id
    if provenance:
        kw["provenance"] = provenance
    return LearnedAIGCProbe(joblib_path, **kw)


OPTIONAL_CHANNELS = {
    "learned_aigc_probe": {
        "factory": _aigc_factory,
        "joblib": WS / "artifacts/p5_full/probe.joblib",
        "description": "frozen CLIP ViT-L/14 linear probe (AIGC, multi-family)",
    },
    "learned_face_probe": {
        "factory": lambda p: _face_factory(p, "learned_face_probe"),
        "joblib": WS / "artifacts/p3_full/face_probe.joblib",
        "description": "YuNet+CLIP face head (CDF+Wild multi-domain, court default)",
    },
    "learned_ffpp_probe": {
        "factory": lambda p: _face_factory(p, "learned_ffpp_probe",
                                           provenance="FF++ c23 (FaceForensics++)"),
        "joblib": WS / "artifacts/p3_ffpp/ffpp_only_probe.joblib",
        "description": "YuNet+CLIP FF++-c23 specialist face head (D-024 channel)",
    },
    "learned_mj_probe": {
        "factory": lambda p: _aigc_factory(p, "learned_mj_probe",
                                           provenance="Midjourney v5 fakes vs external reals"),
        "joblib": WS / "artifacts/p7_mj/probe.joblib",
        "description": "CLIP Midjourney-v5 specialist head (D-026 follow-up channel)",
    },
}


def optional_ids():
    """Registered channels whose joblib actually exists on disk."""
    return [k for k, v in OPTIONAL_CHANNELS.items() if v["joblib"].exists()]


def build(spec: str):
    """spec: 'core' (no optional), 'all', or comma-separated channel ids.

    Returns (modules, ids) for the OPTIONAL channels only — the 5 core
    modules come from the Argus_Master registry and are always prepended
    by the caller. Channels with missing joblibs are skipped with a warning.
    """
    if spec == "core":
        want = []
    elif spec in ("all", ""):
        want = optional_ids()
    else:
        want = [s.strip() for s in spec.split(",") if s.strip()]
    modules, ids = [], []
    for cid in want:
        entry = OPTIONAL_CHANNELS.get(cid)
        if entry is None:
            print(f"[channels] unknown channel '{cid}' — skipped", flush=True)
            continue
        if not entry["joblib"].exists():
            print(f"[channels] {cid}: joblib missing ({entry['joblib']}) — skipped",
                  flush=True)
            continue
        modules.append(entry["factory"](str(entry["joblib"])))
        ids.append(cid)
    return modules, ids


def active_ids() -> list:
    """Full module-id list (core + available optional) — stacker default."""
    return CORE_IDS + optional_ids()
