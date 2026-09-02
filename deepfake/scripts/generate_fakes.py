"""Generate the P0 fake corpus locally on Apple Silicon (MPS) with SD 1.5.

COCO-caption-style prompts (de-confounding: matches real-corpus content),
512px, 25 steps, deterministic seeds, resumable. Output:
  <workspace>/data/corpora/eval_fake/*.png + manifest.json
"""
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

OUT = Path(__file__).resolve().parents[2] / "data" / "corpora" / "eval_fake"
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
N = int(__import__("os").environ.get("ARGUS_FAKE_N", "150"))
STEPS = int(__import__("os").environ.get("ARGUS_FAKE_STEPS", "20"))

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


def prompts(n: int):
    out = []
    for i in range(n):
        s = SUBJECTS[i % len(SUBJECTS)]
        sc = SCENES[(i // len(SUBJECTS)) % len(SCENES)]
        out.append(f"{s} {sc}{SUFFIX}")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    t0 = time.time()
    done = 0
    for i, prompt in enumerate(prompts(N)):
        dest = OUT / f"fake_{i:04d}.png"
        if dest.exists():
            continue
        gen = torch.Generator("mps").manual_seed(10_000 + i)
        img = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=STEPS,
                   guidance_scale=7.5, height=512, width=512, generator=gen).images[0]
        img.save(dest)
        manifest[dest.name] = {"prompt": prompt, "seed": 10_000 + i, "model": MODEL_ID,
                               "steps": STEPS, "guidance": 7.5}
        done += 1
        if done % 10 == 0:
            manifest_path.write_text(json.dumps(manifest, indent=2))
            rate = done / (time.time() - t0) * 60
            print(f"[{i+1}/{N}] {rate:.1f} img/min", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"done: {len(list(OUT.glob('*.png')))} fakes in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
