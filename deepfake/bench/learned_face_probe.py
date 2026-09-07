"""learned_face_probe - supervised face-manipulation head as an Argus evidence module.

Contract-compliant (polarity: negative=fake). Detects the largest face via YuNet,
crops 256px, embeds frozen CLIP ViT-L/14, applies the trained linear probe.
unavailable_reason="no_face" when detection fails (fail-closed per contract).
"""
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
BACKEND = WS / "integrations" / "Argus_Master" / "backend"
for p in (str(BACKEND), str(Path(__file__).parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
from base import EvidenceModule  # noqa: E402
from schemas import Artifact, DegradationState, ModuleOutput  # noqa: E402
from learned_aigc_probe import _ClipService, _MEAN, _STD  # noqa: E402

_YUNET = WS / "tools" / "models" / "yunet.onnx"


def crop_largest_face(pil):
    img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    if min(img.shape[:2]) < 200:
        f = max(1.0, 260 / min(img.shape[:2]))
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(str(_YUNET), "", (w, h), score_threshold=0.55)
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    x, y, fw, fh = [int(v) for v in max(faces, key=lambda t: t[2] * t[3])[:4]]
    pad = int(0.30 * max(fw, fh))
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + fw + pad), min(h, y + fh + pad)
    face = img[y0:y1, x0:x1]
    side = min(face.shape[:2])
    ys, xs = (face.shape[0] - side) // 2, (face.shape[1] - side) // 2
    return cv2.cvtColor(cv2.resize(face[ys:ys + side, xs:xs + side], (256, 256)), cv2.COLOR_BGR2RGB)


class LearnedFaceProbe(EvidenceModule):
    module_id = "learned_face_probe"
    version = "0.1.0"

    def __init__(self, probe_path, module_id=None, provenance="blended-face pseudo-fakes"):
        """module_id/provenance: override when the same architecture serves a
        different trained head (e.g. learned_ffpp_probe trained on FF++ c23)."""
        import joblib
        self.probe = joblib.load(probe_path)
        if module_id:
            self.module_id = module_id
        self.provenance = provenance

    def assess(self, ctx, d: DegradationState, base_reliability: float) -> ModuleOutput:
        face = crop_largest_face(ctx.pil)
        if face is None:
            return self._unavailable("no_face")
        from PIL import Image as _Image
        face224 = _Image.fromarray(face).resize((224, 224), _Image.LANCZOS)
        model, torch, device = _ClipService.get()
        arr = np.asarray(face224, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(x).squeeze(0).float().cpu().numpy()
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        p_fake = float(self.probe.predict_proba(feat.reshape(1, -1))[0, 1])
        e = float(np.clip((0.5 - p_fake) * 2.0, -1.0, 1.0))
        direction = "synthetic" if e < -0.15 else ("authentic" if e > 0.15 else "neutral")
        return ModuleOutput(
            module_id=self.module_id,
            version=self.version,
            evidence_score=round(e, 4),
            reliability_score=base_reliability,
            confidence_score=round(min(0.95, 0.5 + abs(p_fake - 0.5)), 4),
            verdict_direction=direction,
            artifacts=[Artifact(
                type="learned_face_probe_score",
                description=f"Face-crop CLIP probe: P(manipulated)={p_fake:.3f}",
                strength=min(1.0, abs(e) + 0.15),
                checkable_claim=(f"Largest detected face cropped and scored by frozen "
                                 f"CLIP ViT-L/14 linear head trained on {self.provenance}: "
                                 f"P(fake)={p_fake:.3f}"),
            )],
        )
