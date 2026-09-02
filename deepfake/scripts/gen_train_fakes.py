"""Generate TRAINING fakes (disjoint from eval_fake: seeds >=50000, wider param grid).

Varies steps/guidance so the probe learns robust diffusion artifacts rather than
one pipeline fingerprint. Resumable. Output: data/corpora/train_fake + manifest.json
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

OUT = Path(__file__).resolve().parents[2] / "data" / "corpora" / "train_fake"
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
N = int(os.environ.get("ARGUS_TRAINFAKE_N", "250"))

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
    "a pizza with pepperoni and basil on a metal tray", "an elderly man reading a newspaper on a bench",
    "a stop sign at a suburban intersection", "two sheep standing on a hillside",
    "a bowl of soup with bread on a checkered tablecloth", "a skier going down a snowy slope",
    "a motorcycle parked on a narrow street", "a flock of pigeons on a plaza",
    "a tennis player serving on a clay court", "a refrigerator and stove in a small kitchen",
]
SCENES = ["in the morning", "in the afternoon", "at midday", "on an overcast day",
          "in the evening", "in bright sunlight", "on a cloudy day", "at dusk",
          "in soft indoor light", "under clear skies"]
SUFFIX = ", photorealistic photograph, natural lighting"
NEGATIVE = "cartoon, illustration, painting, drawing, blurry, low quality, watermark, text"
STEPS_GRID = [20, 25, 30]
GUIDE_GRID = [6.5, 7.5, 8.5]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    t0 = time.time()
    done = 0
    for i in range(N):
        steps = STEPS_GRID[i % len(STEPS_GRID)]
        guide = GUIDE_GRID[(i // len(STEPS_GRID)) % len(GUIDE_GRID)]
        prompt = f"{SUBJECTS[i % len(SUBJECTS)]} {SCENES[(i // 7) % len(SCENES)]}{SUFFIX}"
        dest = OUT / f"tfake_{i:04d}.png"
        if dest.exists():
            continue
        gen = torch.Generator("mps").manual_seed(50_000 + i)
        img = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=steps,
                   guidance_scale=guide, height=512, width=512, generator=gen).images[0]
        img.save(dest)
        manifest[dest.name] = {"prompt": prompt, "seed": 50_000 + i, "model": MODEL_ID,
                               "steps": steps, "guidance": guide}
        done += 1
        if done % 10 == 0:
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"[{i+1}/{N}] {done/(time.time()-t0)*60:.1f}/min", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"train fakes done: {len(list(OUT.glob('*.png')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
