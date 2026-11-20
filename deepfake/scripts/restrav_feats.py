"""ReStraV 21-D geometry feature extractor, adapted from
github.com/ChristianInterno/ReStraV (NeurIPS 2025, Apache-2.0).

Reads MP4s, samples 24 frames over a 2 s center window, encodes with
DINOv2 ViT-S/14, computes the 21-D (7 distances + 6 angles + 8 stats) feature.
"""
import os, sys, glob, math, warnings, time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from torchvision.transforms import InterpolationMode
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_regular_timestamps

warnings.filterwarnings("ignore", message="xFormers is available.*", category=UserWarning, module=r"dinov2\..*")

_DINOV2 = None

def dinov2():
    global _DINOV2
    if _DINOV2 is None:
        _DINOV2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
    return _DINOV2

def preprocess(frames):
    frames = frames.float() / 255.0
    return torch.stack([T.resize(x, [224, 224], InterpolationMode.BICUBIC, antialias=True) for x in frames])

def decode_clip(video_path, T=24, window_sec=2.0):
    dec = VideoDecoder(video_path)
    dur = float(getattr(dec.metadata, "duration_seconds", 0.0) or 0.0)
    if dur <= 0:
        dur = 2.0
    center = dur / 2.0
    half = window_sec / 2
    start = max(0.0, center - half)
    end = min(dur, center + half)
    delta = window_sec / max(T - 1, 1)
    clip = clips_at_regular_timestamps(
        dec, num_frames_per_clip=T, seconds_between_frames=delta,
        seconds_between_clip_starts=1e9, sampling_range_start=start,
        sampling_range_end=end, policy="repeat_last").data[0]
    if clip.shape[1] not in (1, 3):
        clip = clip.permute(0, 3, 1, 2)
    return clip

def extract_dinov2_embeddings(video_paths, device=None, T=24, window_sec=2.0):
    device = device or torch.device("cpu")
    model = dinov2().to(device)
    outs, keep = [], []
    for p in video_paths:
        try:
            clip = decode_clip(p, T=T, window_sec=window_sec)
            outs.append(preprocess(clip))
            keep.append(p)
        except Exception as e:
            print(f"  decode fail {os.path.basename(p)}: {e}", file=sys.stderr)
    if not outs:
        return None, []
    batch = torch.cat(outs, dim=0).to(device)
    with torch.no_grad():
        feats = model.forward_features(batch)
        cls = feats["x_norm_clstoken"].unsqueeze(1)
        patches = feats["x_norm_patchtokens"]
        tokens = torch.cat([cls, patches], dim=1)
        Z_flat = tokens.flatten(1)
        Z = Z_flat.view(len(outs), outs[0].shape[0], -1)
    return Z, keep

def compute_temporal_geometry(Z):
    delta = Z[:, 1:, :] - Z[:, :-1, :]
    d = delta.norm(dim=-1)
    cos = F.cosine_similarity(delta[:, :-1, :], delta[:, 1:, :], dim=-1)
    theta = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    return d, theta

def moment4(x):
    return x.mean(dim=-1), x.amin(dim=-1), x.amax(dim=-1), x.var(dim=-1, unbiased=False)

def features_from_Z(Z):
    d, t = compute_temporal_geometry(Z)
    d7 = d[:, :7]
    t6 = t[:, :6]
    mu_d, mn_d, mx_d, var_d = moment4(d)
    mu_t, mn_t, mx_t, var_t = moment4(t)
    stats = torch.stack([mu_d, mn_d, mx_d, var_d, mu_t, mn_t, mx_t, var_t], dim=1)
    return torch.cat([d7, t6, stats], dim=1)  # [N, 21]

FEATURE_NAMES = (
    [f"d{i}" for i in range(7)]
    + [f"theta{i}" for i in range(6)]
    + ["mu_d", "min_d", "max_d", "var_d", "mu_t", "min_t", "max_t", "var_t"]
)

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="glob for real MP4s")
    ap.add_argument("--fake", required=True, help="glob for fake MP4s")
    ap.add_argument("--n", type=int, default=50, help="per class")
    ap.add_argument("--out", default="/tmp/restrav_feats.npz")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    reals = sorted(glob.glob(args.real))[:args.n]
    fakes = sorted(glob.glob(args.fake))[:args.n]
    print(f"real={len(reals)} fake={len(fakes)} device={args.device}")
    t0 = time.time()
    Zr, kr = extract_dinov2_embeddings(reals, device=args.device)
    t1 = time.time()
    Zf, kf = extract_dinov2_embeddings(fakes, device=args.device)
    t2 = time.time()
    print(f"real feats {Zr.shape} in {t1-t0:.1f}s; fake feats {Zf.shape} in {t2-t1:.1f}s")
    Fr = features_from_Z(Zr).cpu().numpy()
    Ff = features_from_Z(Zf).cpu().numpy()
    np.savez(args.out, X_real=Fr, X_fake=Ff, paths_real=kr, paths_fake=kf,
             names=np.array(FEATURE_NAMES))
    print(f"saved {args.out}")
    # quick AUC per-feature (real=positive)
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(Fr)), np.zeros(len(Ff))]
    X = np.vstack([Fr, Ff])
    aucs = [roc_auc_score(y, X[:, i]) for i in range(X.shape[1])]
    best = int(np.argmax(np.abs(np.array(aucs) - 0.5)))
    print(f"best single feature: {FEATURE_NAMES[best]} AUC={aucs[best]:.4f}")
    print(f"mean |AUC-0.5| across 21 features: {np.mean(np.abs(np.array(aucs)-0.5)):.4f}")