"""provenance_probe — C2PA / Content Credentials channel (Layer 3).

Reads the file's C2PA manifest (JUMBF) as-is — provenance lives in the
uploaded bytes, never in decoded pixels. Signals (D-034):

  AI-claimed          digitalSourceType=trainedAlgorithmicMedia, or actions /
                      claim_generator naming a known generator -> strongly synthetic
  camera + valid sig  manifest claims capture device and signature validates
                      against the C2PA trust list -> strongly authentic
  broken provenance   manifest present, signature present but validation
                      errors, or corrupt JUMBF -> mildly suspicious
  unsigned manifest   anyone can write one -> mild suspicion
  no manifest         NEUTRAL (most content today; absence is not evidence)

Polarity: negative evidence = synthetic (court convention, D-029).
Deterministic read via c2pa-python; no training, no model file (config
blob stored as the joblib marker for registry compatibility).
"""
import io
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
BACKEND = WS / "integrations" / "Argus_Master" / "backend"
for p in (str(BACKEND), str(Path(__file__).parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from base import EvidenceModule  # noqa: E402
from schemas import Artifact, DegradationState, ModuleOutput  # noqa: E402

AI_GENERATOR_MARKERS = ("dall-e", "firefly", "stable diffusion", "midjourney",
                        "imagen", "grok", "openai", "runway", "stability ai",
                        "sora", "veo", "leonardo", "krea", "napkin")
CAMERA_LABELS = ("c2pa.hash.data", "stds.exif", "c2pa.capture")


def _manifest_signals(mj: dict) -> dict:
    """Signals from a c2pa.Reader.json() top-level structure (verified against
    c2pa-python 0.37: manifests carry claim_generator_info, assertions,
    signature_info; validation_state/status live at the top level)."""
    active = mj.get("active_manifest")
    am = (mj.get("manifests") or {}).get(active, {}) if active else {}
    blob = json.dumps(mj).lower()
    gen_info = " ".join(
        str(g.get("name", "")) + " " + " ".join(str(v) for k, v in g.items()
                                                if isinstance(g, dict) and k.startswith("org."))
        for g in (am.get("claim_generator_info") or []) if isinstance(g, dict)).lower()
    ai = ("trainedalgorithmicmedia" in blob) or \
        any(mk in gen_info for mk in AI_GENERATOR_MARKERS)
    camera = ("c2pa.hash.data" in blob) or ("stds.exif" in blob) or \
        ("c2pa.capture" in blob)
    state = str(mj.get("validation_state", "")).lower()
    codes = [str(x.get("code", "")) for x in (mj.get("validation_status") or [])
             if isinstance(x, dict)]
    untrusted = any("untrusted" in c for c in codes)
    errors = [c for c in codes if "untrusted" not in c]
    return {"ai": ai, "camera": camera,
            "has_signature": bool(am.get("signature_info")),
            "state": state, "untrusted": untrusted, "errors": errors,
            "claim_generator": gen_info or str(am.get("claim_generator", "")).lower()}


class ProvenanceProbe(EvidenceModule):
    module_id = "provenance_probe"
    version = "0.1.0"

    def __init__(self, config_path: Path, module_id: str | None = None):
        self.config = {}
        try:
            import joblib
            self.config = joblib.load(config_path) or {}
        except Exception:
            self.config = {}
        if module_id:
            self.module_id = module_id
        self.provenance = "c2pa-python manifest read (D-034 Layer 3)"

    def assess(self, ctx, d: DegradationState, base_reliability: float) -> ModuleOutput:
        self._rel = base_reliability
        import c2pa
        # Provenance lives in the ORIGINAL ingested bytes. The harness
        # re-encodes pixels into canonical form before modules run, which
        # strips JUMBF/APP11 — so always prefer the untouched upload.
        raw = getattr(ctx, "original_raw", None) or ctx.raw_bytes
        try:
            reader = c2pa.Reader(io.BytesIO(raw))
            manifest_json = json.loads(reader.json())
        except Exception as e:
            name = type(e).__name__
            if "ManifestNotFound" in name or "no JUMBF" in str(e):
                return self._out(0.0, "neutral", 0.5,
                                 "No C2PA manifest present (neutral).", 0.1,
                                 "C2PA JUMBF scan of the uploaded bytes found no "
                                 "Content Credentials manifest. Neutral evidence: "
                                 "most camera and generated content alike carries "
                                 "no manifest today; absence is not evidence of "
                                 "either origin.")
            return self._out(0.3, "synthetic", 0.7,
                             f"Corrupt C2PA/JUMBF data ({name}).", 0.5,
                             f"C2PA manifest data exists but is unreadable ({name}); "
                             "corrupt provenance structures are a manipulation signal.")

        s = _manifest_signals(manifest_json)
        if s["ai"]:
            desc = (f"C2PA manifest claims AI generation "
                    f"(claimGenerator='{s['claim_generator'][:60]}').")
            claim = ("Content Credentials manifest explicitly declares this "
                     "media as machine-generated (trainedAlgorithmicMedia / "
                     "generator-identified). C2PA manifests are signed by the "
                     "generator itself — strong synthetic evidence.")
            return self._out(-0.9, "synthetic", 0.95, desc, 0.9, claim)
        if s["camera"] and s["state"] == "valid" and not s["untrusted"] and not s["errors"]:
            desc = "C2PA manifest: capture provenance, signature validates."
            claim = ("Content Credentials manifest claims camera capture and "
                     "its signature validates. Signed provenance is the "
                     "strongest available authenticity evidence — though it "
                     "attests the file matches the manifest, not world truth.")
            return self._out(0.9, "authentic", 0.9, desc, 0.9, claim)
        if s["state"] and s["state"] != "valid" or s["errors"]:
            desc = (f"C2PA manifest present but validation failed "
                    f"(state={s['state'] or 'unknown'}"
                    f"{', ' + ', '.join(s['errors']) if s['errors'] else ''}).")
            claim = ("A provenance manifest exists but it does not validate. "
                     "Manifests are self-writable, so claimed provenance with "
                     "broken signing is suspicious.")
            return self._out(0.3, "synthetic", 0.75, desc, 0.5, claim)
        if s["has_signature"] and s["untrusted"]:
            desc = "C2PA manifest signed by an untrusted certificate."
            claim = ("A provenance manifest is present, signed by a certificate "
                     "outside the C2PA trust list; manifests are self-writable, "
                     "so this is weak suspicion only.")
            return self._out(0.2, "synthetic", 0.6, desc, 0.4, claim)
        if s["has_signature"] or s["camera"]:
            desc = "C2PA manifest present with undeclared origin."
            claim = ("A Content Credentials manifest exists but declares "
                     "neither AI generation nor verified capture provenance.")
            return self._out(0.1, "neutral", 0.5, desc, 0.3, claim)
        return self._out(0.0, "neutral", 0.5,
                         "C2PA manifest present with no origin claims.", 0.3,
                         "A Content Credentials manifest exists but declares "
                         "neither AI generation nor capture provenance.")

    def _out(self, e, direction, conf, desc, strength, claim):
        return ModuleOutput(
            module_id=self.module_id, version=self.version,
            evidence_score=e, reliability_score=self._rel,
            confidence_score=conf, verdict_direction=direction,
            artifacts=[Artifact(type="provenance_read", description=desc,
                                strength=strength, checkable_claim=claim)])