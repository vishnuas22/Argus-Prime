"""VLM explanation engine — Layer 1, repurposed (D-031).

Measured result: a 7B VLM interrogated with anomaly questions is NOT a
discriminative channel (physics bank AUC 0.500 — never flags; forensic bank
AUC 0.350 — flags 100% of reals). So the VLM does not vote in court.

Its genuine strength: neutral, evidence-file-grade scene narration. This
module attaches human-readable observations to a verdict AFTER the court
has decided — the 'reasoning in any tough situation' layer.

Runs MLX Qwen2.5-VL-7B-Instruct-4bit locally; answers cached by image hash.
"""
import hashlib
import json
import threading
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
CACHE_FILE = WS / "data" / "caches" / "vlm_explain.jsonl"
MODEL_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
BANK_VERSION = 1

PROMPT = (
    "You are a forensic photo analyst writing an evidence file. Describe this image factually:\n"
    "SCENE: one sentence summary.\n"
    "LIGHTING: shadow directions and light consistency.\n"
    "SUBJECTS: pose, visible hands, eyes, accessories — be specific.\n"
    "TEXT: any visible text and its condition.\n"
    "UNUSUAL: list anything visually noteworthy or anomalous, or 'none observed'.\n"
    "Do NOT speculate about whether the image is synthetic or authentic."
)

_lock = threading.Lock()
_model = None
_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE_FILE.exists():
            for line in CACHE_FILE.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    _cache[rec["key"]] = rec
                except Exception:
                    continue
    return _cache


def _append_cache(key, rec):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from mlx_vlm import load
            model, processor = load(MODEL_ID)
            config = model.config
            ip = getattr(processor, "image_processor", None)
            if ip is not None and hasattr(ip, "max_pixels"):
                ip.max_pixels = 28 * 28 * 560
                ip.min_pixels = 28 * 28 * 56
            _model = (model, processor, config)
    return _model


def explain_image(pil_rgb, save_path: Path | None = None) -> dict:
    """Neutral VLM narration of an image. Returns dict with scene/lighting/
    subjects/text/unusual fields + provenance. Cached by image bytes hash."""
    import tempfile
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    if save_path is not None:
        raw = save_path.read_bytes()
    else:
        import io
        buf = io.BytesIO()
        pil_rgb.save(buf, format="PNG")
        raw = buf.getvalue()
    key = hashlib.sha256(raw).hexdigest()[:24] + f"_v{BANK_VERSION}"

    cache = _load_cache()
    if key in cache:
        return {**cache[key]["result"], "cached": True}

    model, processor, config = _get_model()
    tmp = Path(tempfile.mkdtemp(prefix="argus_vlm_")) / "img.png"
    pil_rgb.save(tmp)

    prompt = apply_chat_template(processor, config, PROMPT, num_images=1)
    t0 = time.time()
    out = generate(model, processor, prompt, image=str(tmp), max_tokens=400, verbose=False)
    text = (out.text if hasattr(out, "text") else str(out)).strip()
    elapsed = round(time.time() - t0, 1)

    def grab(tag):
        for line in text.splitlines():
            s = line.strip().replace("*", "").replace("**", "")
            if s.upper().startswith(tag.upper() + ":"):
                return s[len(tag) + 1:].strip()
        return ""

    result = {
        "scene": grab("SCENE"),
        "lighting": grab("LIGHTING"),
        "subjects": grab("SUBJECTS"),
        "text": grab("TEXT"),
        "unusual": grab("UNUSUAL"),
        "narrative": text[:900],
        "elapsed_s": elapsed,
        "model": MODEL_ID.split("/")[-1],
        "cached": False,
    }
    with _lock:
        cache = _load_cache()
        if key not in cache:
            _append_cache(key, {"key": key, "ts": time.time(), "result": result})
            cache[key] = {"key": key, "result": result}
    return result
