"""Backbone registry + lazy embedders — single source of truth for cache build AND serve.

One module owns (a) what each backbone is (id, preprocessing, dim, pooling) and
(b) how to embed an image with it, so `build_feature_cache.py` (train-side
dumps) and the Stage-B evidence modules (serve) can never diverge (D-044
parity class). Feature vectors are returned RAW (no L2): storage convention is
un-normalized fp16, normalization happens at train/serve consumption time
(lessons 2026-10-12 / train_face_multidomain.py:73,82).

Usage:
    from backbone_embed import BACKBONES, embed, get_embedder, resolve_revision
    vec = embed(pil, "dv2b")
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path(__file__).resolve().parents[2]

# dv3 stays a reserved registry entry until the user accepts Meta's license on
# huggingface.co (HF returns HTTP 403 for the gated repos without that click).
DV3_HF_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DV3_EXIT = (f"backbone 'dv3' is reserved: weights live at {DV3_HF_REPO} "
            f"(repo family facebook/dinov3-*) which requires user license "
            f"acceptance on huggingface.co")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

BACKBONES = {
    "clip": {
        "timm_or_hf_id": "vit_large_patch14_clip_224.openai",
        "loader": "timm",
        "repo": "timm/vit_large_patch14_clip_224.openai",
        "input_size": 224,
        "mean": CLIP_MEAN,
        "std": CLIP_STD,
        "dim": 1024,
        "pooling": "cls_post_ln",
        "note": "frozen CLIP ViT-L/14 (timm, global_pool=token) — the "
                "incumbent baseline; mean/std from build_feature_cache.py:17-19",
    },
    "dv2b": {
        "timm_or_hf_id": "vit_base_patch14_dinov2.lvd142m",
        "loader": "timm",
        "repo": "timm/vit_base_patch14_dinov2.lvd142m",
        "input_size": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "dim": 768,
        "pooling": "cls_post_ln",
        "model_kwargs": {"img_size": 224},
        "note": "DINOv2 ViT-B/14 (timm, class_token + global_pool=token); "
                "timm cfg declares 518 so we instantiate at img_size=224 "
                "(=16x14; pos-embed resized by timm's checkpoint filter, "
                "cos=0.897 vs native-518 features — verified); matches the "
                "CLIP arm's input budget; weights already in the local HF "
                "cache (offline-capable)",
    },
    "dv2l": {
        "timm_or_hf_id": "vit_large_patch14_dinov2.lvd142m",
        "loader": "timm",
        "repo": "timm/vit_large_patch14_dinov2.lvd142m",
        "input_size": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "dim": 1024,
        "pooling": "cls_post_ln",
        "model_kwargs": {"img_size": 224},
        "note": "DINOv2 ViT-L/14 (timm, class_token + global_pool=token), "
                "same img_size=224 / preprocessing contract as dv2b",
    },
    "siglip": {
        "timm_or_hf_id": "google/siglip-base-patch16-224",
        "loader": "hf",
        "repo": "google/siglip-base-patch16-224",
        "input_size": 224,
        "mean": SIGLIP_MEAN,
        "std": SIGLIP_STD,
        "dim": 768,
        "pooling": "attention_pooler_post_ln",
        "note": "SigLIP-base/16 via transformers SiglipVisionModel; "
                "mean/std (0.5,0.5,0.5); pooling verified on the actual "
                "checkpoint: attention-pooler head present (7.09M params, no "
                "MISSING vision keys) -> post-LN pooled image embedding "
                "(attention_pooler_post_ln); falls back to post-LN CLS if a "
                "checkpoint ever ships without the head",
    },
    "dv3": {
        "timm_or_hf_id": DV3_HF_REPO,
        "loader": "reserved",
        "repo": DV3_HF_REPO,
        "input_size": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "dim": 768,
        "pooling": "cls_post_ln",
        "note": "RESERVED — not runnable until the user accepts Meta's license "
                "on huggingface.co; get_embedder('dv3') raises SystemExit",
    },
}

_SERVICE: dict = {}
_POOLING_RESOLVED: dict = {}


def _check(backbone_id: str) -> dict:
    if backbone_id == "dv3":
        raise SystemExit(DV3_EXIT)
    if backbone_id not in BACKBONES:
        raise SystemExit(f"unknown backbone {backbone_id!r} "
                         f"(known: {sorted(BACKBONES)})")
    return BACKBONES[backbone_id]


def resolve_revision(backbone_id: str) -> str | None:
    """HF snapshot sha for the backbone's repo, read from the local HF cache
    (the snapshot dir name IS the resolved revision). None when offline/unresolved."""
    cfg = _check(backbone_id)
    repo = cfg.get("repo")
    if not repo:
        return None
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        hub = Path(HF_HUB_CACHE)
    except Exception:
        hub = Path(os.environ.get("HF_HUB_CACHE",
                                  Path.home() / ".cache/huggingface/hub"))
    org, name = repo.split("/", 1)
    snaps = hub / f"models--{org}--{name}" / "snapshots"
    if not snaps.is_dir():
        return None
    shas = sorted(p.name for p in snaps.iterdir() if p.is_dir())
    return shas[-1] if shas else None


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_embedder(backbone_id: str):
    """Lazy singleton per backbone (mirrors learned_aigc_probe._ClipService).

    Returns a callable pil -> raw np.float32 feature vector (no L2).
    """
    if backbone_id == "dv3":
        raise SystemExit(DV3_EXIT)
    if backbone_id in _SERVICE:
        return _SERVICE[backbone_id]
    cfg = _check(backbone_id)
    device = _device()
    mean = np.array(cfg["mean"], dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)
    size = int(cfg["input_size"])

    if cfg["loader"] == "timm":
        import timm
        import torch
        model = timm.create_model(cfg["timm_or_hf_id"], pretrained=True,
                                  num_classes=0,
                                  **cfg.get("model_kwargs", {})).to(device).eval()

        def fn(pil: Image.Image) -> np.ndarray:
            arr = np.asarray(pil.convert("RGB").resize((size, size), Image.LANCZOS),
                             dtype=np.float32) / 255.0
            arr = (arr - mean) / std
            x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model(x).squeeze(0).float().cpu().numpy()
            return np.asarray(f, dtype=np.float32).reshape(-1)

    elif cfg["loader"] == "hf":
        import torch
        from transformers import SiglipVisionModel
        model = SiglipVisionModel.from_pretrained(cfg["timm_or_hf_id"])
        model = model.to(device).eval()
        use_head = bool(getattr(model, "use_head", False))
        _POOLING_RESOLVED[backbone_id] = ("attention_pooler_post_ln" if use_head
                                          else "cls_post_ln")

        def fn(pil: Image.Image) -> np.ndarray:
            arr = np.asarray(pil.convert("RGB").resize((size, size), Image.LANCZOS),
                             dtype=np.float32) / 255.0
            arr = (arr - mean) / std
            x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(pixel_values=x)
            f = out.pooler_output
            if f is None:  # checkpoint ships without the attention pooler head
                f = out.last_hidden_state[:, 0, :]
            return np.asarray(f.squeeze(0).float().cpu().numpy(),
                              dtype=np.float32).reshape(-1)
    else:
        raise SystemExit(DV3_EXIT)

    dim = int(cfg["dim"])
    _SERVICE[backbone_id] = fn
    return fn


def embed(pil: Image.Image, backbone_id: str) -> np.ndarray:
    """Raw (un-normalized) feature vector for one image."""
    return get_embedder(backbone_id)(pil)


def embed_dim(backbone_id: str) -> int:
    return int(_check(backbone_id)["dim"])


def entry(backbone_id: str) -> dict:
    """Registry entry (copy) with the runtime-resolved pooling for HF models."""
    cfg = dict(_check(backbone_id))
    if backbone_id in _POOLING_RESOLVED:
        cfg["pooling"] = _POOLING_RESOLVED[backbone_id]
    return cfg


if __name__ == "__main__":
    bid = sys.argv[1] if len(sys.argv) > 1 else "dv2b"
    cfg = entry(bid)
    print(bid, "->", {k: cfg[k] for k in
                      ("timm_or_hf_id", "input_size", "dim", "pooling")})
    print("revision:", resolve_revision(bid))
    from PIL import Image as _I
    img = _I.new("RGB", (320, 240), (120, 40, 200))
    v = embed(img, bid)
    print("embed:", v.shape, v.dtype, "finite:", bool(np.isfinite(v).all()))
