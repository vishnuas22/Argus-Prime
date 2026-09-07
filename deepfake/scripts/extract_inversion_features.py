"""Extract SDXL-VAE inversion features for image corpora (DIRE/AEROBLADE-style).

Real photos survive a VAE encode->decode round-trip poorly (sensor noise destroyed,
texture hallucinated); generated images lie ON the decoder manifold and reconstruct
almost perfectly. We cache a small feature vector per image for a linear probe.

Usage:
  .venv/bin/python deepfake/scripts/extract_inversion_features.py OUT.npz DIR[:label] [DIR[:label] ...]
  label 0 = real, 1 = fake
"""
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SIZE = 512
BATCH = 8


def load_vae():
    import torch
    from diffusers import AutoencoderKL
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae",
        local_files_only=True, torch_dtype=torch.float32,
    ).to(device).eval()
    return vae, torch, device


def to_tensor(pil, torch, device):
    from PIL import Image
    x = np.asarray(pil.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS), np.float32) / 255.0
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)


def features(x, recon, torch, device):
    """x, recon: (1,3,H,W) in [0,1] on device. Returns 7-dim np feature."""
    err = (x - recon).abs().squeeze(0)              # (3,H,W)
    mse_map = err.mean(0)                            # (H,W)

    # global error
    mse = float(mse_map.mean())

    # simple global SSIM (uniform 7x7 window, C1/C2 on L=1)
    pad = 3
    mu_x = torch.nn.functional.avg_pool2d(x.mean(1, keepdim=True), 7, 1, pad)
    mu_r = torch.nn.functional.avg_pool2d(recon.mean(1, keepdim=True), 7, 1, pad)
    vx = torch.nn.functional.avg_pool2d((x.mean(1, keepdim=True) - mu_x) ** 2, 7, 1, pad)
    vr = torch.nn.functional.avg_pool2d((recon.mean(1, keepdim=True) - mu_r) ** 2, 7, 1, pad)
    cov = torch.nn.functional.avg_pool2d(
        (x.mean(1, keepdim=True) - mu_x) * (recon.mean(1, keepdim=True) - mu_r), 7, 1, pad)
    ssim_map = ((2 * mu_x * mu_r + 0.01) * (2 * cov + 0.03)) / ((mu_x**2 + mu_r**2 + 0.01) * (vx + vr + 0.03))
    ssim = float(1.0 - ssim_map.mean())              # DSSIM-style distance

    # high-frequency error: reals have sensor noise the VAE removes
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device).view(1, 1, 3, 3)
    lap_x = torch.nn.functional.conv2d(x.mean(1, keepdim=True), k, padding=1).abs().mean()
    lap_r = torch.nn.functional.conv2d(recon.mean(1, keepdim=True), k, padding=1).abs().mean()
    hf = float(lap_x - lap_r)                        # texture lost in round-trip

    # 8x8 block MSE dispersion
    blocks = mse_map.view(8, 64, 8, 64).mean(dim=(1, 3)).flatten()
    return [mse, ssim, hf, float(blocks.std()), float(blocks.quantile(0.9)), float(blocks.max()), float(err.max())]


def main():
    out_npz = Path(sys.argv[1])
    feats, labels, paths = [], [], []
    vae, torch, device = load_vae()

    pending_pils, pending_meta = [], []

    def flush():
        if not pending_pils:
            return
        batch = torch.cat(pending_pils, 0)
        with torch.no_grad():
            lat = vae.encode(batch).latent_dist.mode()
            recon = vae.decode(lat).sample.clamp(0, 1)
        for i, (f, lab, p) in enumerate(pending_meta):
            feats.append(features(batch[i:i+1], recon[i:i+1], torch, device))
            labels.append(lab)
            paths.append(p)
        pending_pils.clear(); pending_meta.clear()

    for spec in sys.argv[2:]:
        d, lab = spec.rsplit(":", 1)
        d = Path(d)
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in EXTS)
        print(f"[{d}] label={lab} n={len(files)}", flush=True)
        for i, p in enumerate(files):
            from PIL import Image
            try:
                pil = Image.open(p)
                pending_pils.append(to_tensor(pil, torch, device))
                pending_meta.append((None, int(lab), str(p)))
            except Exception as e:
                print(f"  skip {p.name}: {e}", flush=True)
                continue
            if len(pending_pils) >= BATCH:
                flush()
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)
    flush()

    F = np.array(feats, np.float32)
    np.savez(out_npz, features=F, labels=np.array(labels), paths=np.array(paths))
    print(f"saved {F.shape} -> {out_npz}", flush=True)


if __name__ == "__main__":
    main()