"""Read-only C2PA transparency validator (P9, spec p9-c2pa-readonly-v1).

Standalone library + CLI. This module never influences detection: it is not
imported by the court, registers no channel, performs no network or trust-store
access, and writes nothing.

Run:
  ./.venv/bin/python deepfake/scripts/c2pa_read.py PATH [--max-bytes N] [--pretty]
"""
import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import c2pa

SCHEMA_VERSION = "p9.c2pa_read.v1"
MAX_BYTES = 134_217_728
MANIFEST_STATES = ("absent", "well_formed_valid", "trusted", "invalid", "error")

AI_GENERATOR_MARKERS = ("dall-e", "firefly", "stable diffusion", "midjourney",
                        "imagen", "grok", "openai", "runway", "stability ai",
                        "sora", "veo", "leonardo", "krea", "napkin")
CAMERA_MARKERS = ("digitalcapture", "compositecapture", "c2pa.captured",
                  "stds.exif", "c2pa.capture")
ERROR_PREFIXES = ("file_not_found", "file_unreadable", "file_too_large",
                  "invalid_max_bytes", "reader_error", "json_error",
                  "invalid_manifest_json", "internal_error")
NO_MANIFEST_MARKERS = ("manifestnotfound", "no jumbf", "manifest not found")


def _truncate(text: str, limit: int = 500) -> str:
    return text[:limit]


def _normalize_error(error: Any) -> str:
    text = str(error).strip()
    if not text:
        return "internal_error: unspecified error"
    prefix = text.split(":", 1)[0].strip()
    if prefix not in ERROR_PREFIXES:
        text = f"internal_error: {text}"
    return _truncate(text)


def _empty_result(path: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "path": path,
        "manifest_state": "absent",
        "trusted": False,
        "hard_binding": False,
        "ai_claimed": False,
        "camera_claimed": False,
        "signer_issuer": None,
        "validation_state": None,
        "validation_status_codes": [],
        "error": None,
    }


def _error_result(path: str, prefix: str, detail: str = "") -> dict:
    result = _empty_result(path)
    result["manifest_state"] = "error"
    text = prefix if not detail else f"{prefix}: {detail}"
    result["error"] = _truncate(text)
    return result


def _code_strings(value: Any) -> list[str]:
    items = value if isinstance(value, (list, tuple)) else [value]
    codes = []
    for item in items:
        if isinstance(item, dict):
            code = item.get("code")
        elif isinstance(item, str):
            code = item
        else:
            code = None
        if isinstance(code, str):
            code = code.strip()
            if code:
                codes.append(code)
    return codes


def _active_manifest(manifest: dict) -> tuple[dict, dict, Any]:
    manifests = manifest.get("manifests")
    if not isinstance(manifests, dict):
        manifests = {}
    active_label = manifest.get("active_manifest")
    if isinstance(active_label, str) and active_label in manifests:
        active = manifests[active_label]
    elif len(manifests) == 1:
        active = next(iter(manifests.values()))
    else:
        active = {}
    if not isinstance(active, dict):
        active = {}
    return active, manifests, active_label


def _active_codes(manifest: dict, active: dict, key: str) -> list[str]:
    codes = _code_strings(active.get(key))
    validation_results = manifest.get("validation_results")
    if isinstance(validation_results, dict):
        active_results = validation_results.get("activeManifest")
        if isinstance(active_results, dict):
            codes.extend(_code_strings(active_results.get(key)))
    return codes


def _claim_generator_text(active: dict) -> str:
    parts = []
    for key in ("claim_generator_info", "claim_generator"):
        value = active.get(key)
        if value is not None:
            parts.append(json.dumps(value, sort_keys=True))
    return " ".join(parts).lower()


def classify_manifest(manifest: dict | None, *, path: str = "<memory>",
                      error: str | None = None) -> dict:
    if error is not None:
        result = _empty_result(path)
        result["manifest_state"] = "error"
        result["error"] = _normalize_error(error)
        return result
    if manifest is None:
        return _empty_result(path)
    if not isinstance(manifest, dict):
        return _error_result(path, "invalid_manifest_json",
                             "manifest JSON is not an object")

    raw_state = manifest.get("validation_state")
    raw = raw_state.strip().lower() if isinstance(raw_state, str) else ""
    valid_raw = raw in {"valid", "trusted"}

    active, manifests, active_label = _active_manifest(manifest)
    has_active = (isinstance(active_label, str) and bool(active_label)
                  and active_label in manifests) or len(manifests) == 1

    success = _active_codes(manifest, active, "success")
    failure = _active_codes(manifest, active, "failure")
    codes = sorted(set(_code_strings(manifest.get("validation_status")) + failure))

    sig = active.get("signature_info")
    has_sig = isinstance(sig, dict) and bool(sig)

    if not has_active:
        codes = sorted(set(codes + ["manifest.activeManifest.missing"]))
        state = "invalid"
    elif not valid_raw or any("untrusted" not in c.lower() for c in codes):
        state = "invalid"
    elif (codes == [] and "signingCredential.trusted" in success and has_sig):
        state = "trusted"
    else:
        state = "well_formed_valid"

    hard_binding = bool(
        valid_raw
        and (("assertion.dataHash.match" in success)
             or ("assertion.bmffHash.match" in success))
        and not any(("assertion.dataHash.mismatch" in c)
                    or ("assertion.bmffHash.mismatch" in c) for c in codes)
    )

    blob = json.dumps(active, sort_keys=True).lower()
    generator_text = _claim_generator_text(active)
    ai_claimed = (
        "trainedalgorithmicmedia" in blob
        or "compositesynthetic" in blob
        or any(marker in generator_text for marker in AI_GENERATOR_MARKERS)
    )
    camera_claimed = any(marker in blob for marker in CAMERA_MARKERS)

    signer_issuer = None
    if isinstance(sig, dict):
        issuer = sig.get("issuer")
        if isinstance(issuer, str) and issuer.strip():
            signer_issuer = issuer.strip()

    result = _empty_result(path)
    result["manifest_state"] = state
    result["trusted"] = state == "trusted"
    result["hard_binding"] = hard_binding
    result["ai_claimed"] = bool(ai_claimed)
    result["camera_claimed"] = bool(camera_claimed)
    result["signer_issuer"] = signer_issuer
    result["validation_state"] = raw_state if isinstance(raw_state, str) else None
    result["validation_status_codes"] = codes
    result["error"] = None
    return result


def _absolute_path(path: str | Path) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError, TypeError):
        return str(path)


def _valid_max_bytes(max_bytes: Any) -> bool:
    return (isinstance(max_bytes, int) and not isinstance(max_bytes, bool)
            and max_bytes > 0)


def _is_no_manifest(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return any(marker in name or marker in text for marker in NO_MANIFEST_MARKERS)


def _exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".strip()


def read_c2pa(path: str | Path, *, max_bytes: int = MAX_BYTES) -> dict:
    label = _absolute_path(path)
    if not _valid_max_bytes(max_bytes):
        return _error_result(label, "invalid_max_bytes",
                             f"max_bytes must be a positive integer, got {max_bytes!r}")

    source = Path(path).expanduser()
    if not source.exists():
        return _error_result(label, "file_not_found", f"{source} does not exist")
    if not source.is_file():
        return _error_result(label, "file_unreadable",
                             f"{source} is not a regular file")

    try:
        size = source.stat().st_size
    except OSError as exc:
        return _error_result(label, "file_unreadable", _exception_text(exc))
    if size > max_bytes:
        return _error_result(label, "file_too_large",
                             f"{size} bytes exceeds max_bytes={max_bytes}")

    try:
        with source.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        return _error_result(label, "file_unreadable", _exception_text(exc))
    if len(raw) > max_bytes:
        return _error_result(label, "file_too_large",
                             f"{len(raw)} bytes exceeds max_bytes={max_bytes}")

    try:
        reader = c2pa.Reader.try_create(stream=io.BytesIO(raw))
    except Exception as exc:
        if _is_no_manifest(exc):
            return _empty_result(label)
        return _error_result(label, "reader_error", _exception_text(exc))
    if reader is None:
        return _empty_result(label)

    try:
        try:
            payload = reader.json()
        except Exception as exc:
            return _error_result(label, "json_error", _exception_text(exc))
        try:
            manifest = json.loads(payload)
        except Exception as exc:
            return _error_result(label, "json_error", _exception_text(exc))
        if not isinstance(manifest, dict):
            return _error_result(label, "invalid_manifest_json",
                                 "manifest JSON is not an object")
        try:
            return classify_manifest(manifest, path=label)
        except Exception as exc:
            return _error_result(label, "internal_error", _exception_text(exc))
    finally:
        try:
            reader.close()
        except Exception:
            pass


def _exit_code(result: dict) -> int:
    error = result.get("error")
    if not error:
        return 0
    prefix = error.split(":", 1)[0].strip()
    if prefix in {"file_not_found", "file_unreadable", "file_too_large",
                  "invalid_max_bytes"}:
        return 2
    if prefix == "internal_error":
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="c2pa_read.py",
        description="Read-only C2PA transparency validator.",
    )
    parser.add_argument("path", help="file to inspect")
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES,
                        help=f"maximum bytes to read (default {MAX_BYTES})")
    parser.add_argument("--pretty", action="store_true",
                        help="pretty-print JSON with indent=2")
    args = parser.parse_args(argv)

    try:
        result = read_c2pa(args.path, max_bytes=args.max_bytes)
    except Exception as exc:
        result = _error_result(_absolute_path(args.path), "internal_error",
                               _exception_text(exc))

    if args.pretty:
        output = json.dumps(result, indent=2, sort_keys=True)
    else:
        output = json.dumps(result, sort_keys=True, separators=(",", ":"))
    print(output)
    if result.get("error"):
        print(f"c2pa_read: {result['error']}", file=sys.stderr)
    return _exit_code(result)


if __name__ == "__main__":
    sys.exit(main())
