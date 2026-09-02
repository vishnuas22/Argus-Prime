"""Generate CROSS-GENERATOR fake corpora with architecturally diverse models.

Each output dir is an unseen-generator eval set for trained probes:
  xgen_wuerstchen  (latent cascade)      warp-ai/wuerstchen
  xgen_pixart      (DiT transformer)     PixArt-alpha/PixArt-XL-2-1024
  xgen_sdxl        (large UNet/new VAE)  stabilityai/stable-diffusion-xl-base-1.0

Resumable; manifest per dir; models generated sequentially (MPS is single-tenant).
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2] / "data" / "corpora"
N = int(os.environ.get("ARGUS_XGEN_N", "40"))

SUBJECTS = [
    "a man riding a skateboard down a city street", "two dogs playing with a frisbee in a park",
    "a plate of spaghetti with tomato sauce on a wooden table", "a woman holding an umbrella at a bus stop",
    "a group of people eating pizza around a table", "a brown horse grazing in a fenced field",
    "a kitchen counter with a bowl of fruit and a coffee maker", "a man surfing a small wave at the beach",
    "a double-decker bus driving past shops", "a cat sleeping on a windowsill next to a potted plant",
    "a baseball player swinging a bat", "a bathroom sink with a mirror and folded towels",
    "several boats docked in a small harbor", "a child flying a colorful kite in a field",
    "a train arriving at an outdoor station platform", "a vendor selling vegetables at a street market",
    "a living room with a grey couch and bookshelf", "a cyclist riding across a bridge",
    "a giraffe standing near trees in a zoo", "a laptop notebook and coffee on an office desk",
]
SUFFIX = ", photorealistic photograph, natural lighting"
NEGATIVE = "cartoon, illustration, painting, drawing, blurry, low quality, watermark, text"


def prompt_for(i):
    return f"{SUBJECTS[i % len(SUBJECTS)]}{SUFFIX}"


def load_pixart():
    from diffusers import PixArtAlphaPipeline
    return PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024",
                                               torch_dtype=torch.float16), {"height": 1024, "width": 1024}


def load_sana():
    from diffusers import SanaPipeline
    return SanaPipeline.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_diffusers",
                                        torch_dtype=torch.float16), {"height": 1024, "width": 1024}


def load_sdxl():
    from diffusers import StableDiffusionXLPipeline
    return StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                     torch_dtype=torch.float16,
                                                     variant="fp16"), {"height": 512, "width": 512}


MODELS = {
    "xgen_pixart": ("PixArt-alpha/PixArt-XL-2-1024", load_pixart),
    "xgen_sana": ("Efficient-Large-Model/Sana_1600M_1024px_diffusers", load_sana),
    "xgen_sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", load_sdxl),
}


def run_one(tag):
    model_id, loader = MODELS[tag]
    out = ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    pipe, extra = loader()
    pipe = pipe.to("mps")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    t0 = time.time()
    done = 0
    for i in range(N):
        dest = out / f"{tag}_{i:03d}.png"
        if dest.exists():
            continue
        gen = torch.Generator("mps").manual_seed(90_000 + i)
        if isinstance(pipe, dict):
            prior_out = pipe["prior"](
                prompt=prompt_for(i), negative_prompt=NEGATIVE, generator=gen,
                num_inference_steps=20, guidance_scale=4.0, **extra)
            img = pipe["decoder"](
                image_embeddings=prior_out.image_embeddings.to(torch.float16),
                prompt=prompt_for(i), negative_prompt=NEGATIVE,
                guidance_scale=0.0, num_inference_steps=10,
                generator=torch.Generator("mps").manual_seed(90_000 + i)).images[0]
        else:
            kwargs = dict(prompt=prompt_for(i), negative_prompt=NEGATIVE, generator=gen, **extra)
            img = pipe(**kwargs).images[0]
        img.save(dest)
        manifest[dest.name] = {"prompt": kwargs["prompt"], "seed": 90_000 + i,
                               "model": model_id, **{k: v for k, v in extra.items()}}
        done += 1
        if done % 5 == 0:
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"[{tag}] {done} new, {done/(time.time()-t0)*60:.1f}/min", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{tag}] COMPLETE: {len(list(out.glob('*.png')))} images", flush=True)


def main() -> int:
    tags = sys.argv[1:] or list(MODELS)
    for tag in tags:
        assert tag in MODELS, f"unknown model tag {tag}"
        print(f"=== {tag} ===", flush=True)
        run_one(tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
