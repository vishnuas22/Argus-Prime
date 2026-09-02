"""learned_aigc_probe — supervised CLIP linear probe as an Argus evidence module.

Contract-compliant EvidenceModule (polarity: negative=synthetic). Loads the
trained sklearn probe + a lazily-initialized shared CLIP ViT-L/14 embedder.
"""
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


class _ClipService:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            import timm
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = timm.create_model("vit_large_patch14_clip_224.openai",
                                      pretrained=True, num_classes=0).to(device).eval()
            cls._instance = (model, torch, device)
        return cls._instance


_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def embed(pil):
    model, torch, device = _ClipService.get()
    arr = np.asarray(pil.convert("RGB").resize((224, 224), __import__("PIL.Image", fromlist=["LANCZOS"]).LANCZOS),
                     dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model(x).squeeze(0).float().cpu().numpy()
    return f / (np.linalg.norm(f) + 1e-8)


class LearnedAIGCProbe(EvidenceModule):
    module_id = "learned_aigc_probe"
    version = "0.1.0"

    def __init__(self, probe_path: Path):
        import joblib
        self.probe = joblib.load(probe_path)

    def assess(self, ctx: "ImageContext", d: DegradationState, base_reliability: float) -> ModuleOutput:
        feat = embed(ctx.pil).reshape(1, -1)
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
                type="learned_probe_score",
                description=f"CLIP ViT-L/14 linear probe: P(AI-generated)={p_fake:.3f}",
                strength=min(1.0, abs(e) + 0.15),
                checkable_claim=(f"Frozen CLIP ViT-L/14 features fed to an L2 logistic probe "
                                 f"(UnivFD-style, trained on COCO-reals vs SD1.5-fakes with "
                                 f"jpeg75/resize75 views): P(AI-generated)={p_fake:.3f}"),
            )],
        )
