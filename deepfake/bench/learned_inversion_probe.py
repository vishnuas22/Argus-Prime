"""learned_inversion_probe — VAE round-trip inversion-error channel (Layer 2).

Generator-agnostic principle (DIRE/AEROBLADE): real photos are OFF the
diffusion-VAE decoder manifold (sensor noise destroyed, texture hallucinated
on reconstruction); generated images are ON it and reconstruct almost
perfectly. We compute 7 handcrafted round-trip error features and feed a
logistic probe trained on COCO-reals vs SD-family fakes.

Zero-shot measured (D-029 intake): FLUX 0.981 AUC, MJ 0.849, SG2 9/11 —
on generators never seen in training. Polarity: negative evidence = synthetic.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
BACKEND = WS / "integrations" / "Argus_Master" / "backend"
for p in (str(BACKEND), str(Path(__file__).parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from base import EvidenceModule  # noqa: E402
from schemas import Artifact, DegradationState, ModuleOutput  # noqa: E402

# Reuse the extractor's exact feature code so training/serving never diverge.
_spec = importlib.util.spec_from_file_location(
    "extract_inversion_features",
    WS / "deepfake" / "scripts" / "extract_inversion_features.py")
_ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ext)


class _VaeService:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            vae, torch_m, device_m = _ext.load_vae()
            cls._instance = (vae, torch_m, device_m)
        return cls._instance


class LearnedInversionProbe(EvidenceModule):
    module_id = "learned_inversion_probe"
    version = "0.1.0"

    def __init__(self, probe_path: Path, module_id: str | None = None,
                 provenance: str = "COCO-reals vs SD-family fakes; zero-shot FLUX 0.98/MJ 0.85"):
        import joblib
        self.probe = joblib.load(probe_path)
        if module_id:
            self.module_id = module_id
        self.provenance = provenance

    def assess(self, ctx: "ImageContext", d: DegradationState, base_reliability: float) -> ModuleOutput:
        vae, torch, device = _VaeService.get()
        x = _ext.to_tensor(ctx.pil, torch, device)
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.mode()
            recon = vae.decode(lat).sample.clamp(0, 1)
        feat = np.array(_ext.features(x, recon, torch, device), np.float32).reshape(1, -1)
        p_fake = float(self.probe.predict_proba(feat)[0, 1])
        e = float(np.clip((0.5 - p_fake) * 2.0, -1.0, 1.0))
        direction = "synthetic" if e < -0.15 else ("authentic" if e > 0.15 else "neutral")
        confidence = min(0.95, 0.5 + abs(p_fake - 0.5))
        return ModuleOutput(
            module_id=self.module_id,
            version=self.version,
            evidence_score=round(e, 4),
            reliability_score=base_reliability,
            confidence_score=round(confidence, 4),
            verdict_direction=direction,
            artifacts=[Artifact(
                type="inversion_probe_score",
                description=(f"SD-VAE round-trip error probe: P(synthetic)={p_fake:.3f} "
                             f"(mse={feat[0,0]:.4f}, ssim_d={feat[0,1]:.4f})"),
                strength=min(1.0, abs(e) + 0.15),
                checkable_claim=(f"Image passed through the frozen SD VAE (encode->decode); "
                                 f"reconstruction-error features fed to a logistic probe "
                                 f"({self.provenance}): P(synthetic)={p_fake:.3f}. "
                                 f"Generated images reconstruct near-perfectly (on-manifold); "
                                 f"camera-captured photos do not."),
            )],
        )
