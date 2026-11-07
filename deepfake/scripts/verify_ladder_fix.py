"""2026-11-06 audit F2 fix verification — PRE-REGISTERED asserts (D-046).

Fix-pack under test:
  1. recovery.py audio-ladder issuance gate (_audio_issuable): issue ONLY if
     rescored p >= 0.65 (fake-ward decided) AND |rescored - raw| <= 0.15
     (confirmation). Measured-bias guard: a real-ward-biased rescore can never
     certify REAL.
  2. server.py p_fake_adjudicated / adjudicated response fields (surface
     unification with the UI dial).

Expected values come from the audit's controlled pairs (AUDIT_REPORT.md §3):
  #608 DF-Eval REAL  xj17VCL-uePIstzu6IKU2OMJ1S8.mp3  raw 0.9396 -> was 0.0382
  #610 DF-Eval FAKE  xLo0BGfRoyYdFUhyQfRm9mlEWv4.mp3  raw 0.9897 -> was 0.5609
  #606 imagen4 fake  02be9d8488fa9066.jpg              raw 0.7749 -> 0.8163 (must NOT change)

Run: PATH="$PWD/tools/bin:$PATH" ./.venv/bin/python deepfake/scripts/verify_ladder_fix.py
"""
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
for p in (WS / "deepfake", WS / "deepfake" / "bench", WS / "deepfake" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" | " + str(detail)) if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------- V0: pure gate table ----------------
from recovery import _audio_issuable  # noqa: E402

TABLE = [
    (0.038, 0.9396, False, "rescore_real_ward_bias_guard"),   # #608 shape
    (0.5609, 0.9897, False, "rescore_real_ward_bias_guard"),  # #610 shape
    (0.88, 0.85, True, None),                                  # confirmation rescue
    (0.90, 0.20, False, "rescore_delta_exceeds_cap"),          # delta guard
    (0.6499, 0.60, False, "rescore_real_ward_bias_guard"),     # just under band
    (0.65, 0.50, True, None),                                  # exact band edge
    (0.50, None, False, "rescore_real_ward_bias_guard"),       # null raw -> bias guard only
]
for pf, raw, ok, why in TABLE:
    got = _audio_issuable(pf, raw)
    check(f"V0 issuable({pf}, {raw})", got == (ok, why), f"got {got}")

# ---------------- V1/V2: audit fixtures end-to-end ----------------
from argus_detect import detect_audio  # noqa: E402

DFEVAL = WS / "data" / "dfeval" / "audio" / "audio-data" / "audio-data"

r608 = detect_audio(DFEVAL / "xj17VCL-uePIstzu6IKU2OMJ1S8.mp3")
rec608 = r608.get("recovery") or {}
check("V1 #608 raw p preserved", r608.get("p_fake_audio_mean") == 0.9396,
      r608.get("p_fake_audio_mean"))
check("V1 #608 terminal needs_review", rec608.get("outcome") == "needs_review",
      rec608.get("outcome"))
check("V1 #608 bias-guard reason",
      "rescore_real_ward_bias_guard" in (rec608.get("reasons") or []),
      rec608.get("reasons"))
a2_608 = [s for s in rec608.get("attempted_steps", []) if s.get("step") == "A2"]
print("   #608 A2 receipt:", json.dumps(a2_608[-1]["detail"]) if a2_608 else "none")

r610 = detect_audio(DFEVAL / "xLo0BGfRoyYdFUhyQfRm9mlEWv4.mp3")
rec610 = r610.get("recovery") or {}
check("V2 #610 raw p preserved", r610.get("p_fake_audio_mean") == 0.9897,
      r610.get("p_fake_audio_mean"))
check("V2 #610 terminal needs_review (erosion gone)",
      rec610.get("outcome") == "needs_review", rec610.get("outcome"))
a2_610 = [s for s in rec610.get("attempted_steps", []) if s.get("step") == "A2"]
print("   #610 A2 receipt:", json.dumps(a2_610[-1]["detail"]) if a2_610 else "none")

# ---------------- V3: image ladder regression (#606 must NOT move) ----------------
from PIL import Image  # noqa: E402
from argus_detect import detect_image, _image_stacker_clfs  # noqa: E402

img_path = WS / "data" / "corpora" / "gen_exam" / "imagen4" / "fake" / "02be9d8488fa9066.jpg"
pil = Image.open(img_path)
pil.load()
r606 = detect_image(pil, stacker_clfs=_image_stacker_clfs(),
                    original_raw=img_path.read_bytes())
rec606 = r606.get("recovery") or {}
check("V3 #606 raw p unchanged", r606.get("p_fake_fused") == 0.7749,
      r606.get("p_fake_fused"))
check("V3 #606 still recovered", rec606.get("outcome") == "recovered",
      rec606.get("outcome"))
v606 = (rec606.get("verdict") or {}).get("p_fake_fused")
check("V3 #606 adjudicated 0.8163 byte-equal", v606 == 0.8163, v606)

print()
if FAILS:
    print(f"RESULT: {len(FAILS)} FAIL -> {FAILS}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASS")
