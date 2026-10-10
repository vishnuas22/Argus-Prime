"""End-to-end visual test of the Argus UI: real browser, real user flow.

Stages: load -> health (chip + API contract) -> drop image -> deliberation ->
verdict -> evidence -> drop video (SKIP note when no .mp4 fixture exists).
Captures console errors, page errors, and screenshots at every stage to
artifacts/e2e_ui/.
Usage: .venv/bin/python deepfake/scripts/e2e_ui_test.py <image_path>
"""
import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
OUT = WS / "artifacts" / "e2e_ui"
IMG = Path(sys.argv[1]) if len(sys.argv) > 1 else None

from playwright.sync_api import sync_playwright

issues = []
shots = []


def shot(page, name):
    p = OUT / f"{name}.png"
    page.screenshot(path=str(p), full_page=(name != "01_hero"))
    shots.append(p.name)


def find_video_fixture():
    """Smallest .mp4 between 10KB and 15MB under artifacts/ or data/; None if none.

    Never fabricates a fixture — missing fixture means the video path SKIPs.
    (Lower bound excludes 0-byte placeholder files in data/raw.)"""
    for root in (WS / "artifacts", WS / "data"):
        cands = [p for p in root.rglob("*.mp4")
                 if p.is_file() and 10_000 <= p.stat().st_size <= 15 * 1024 * 1024]
        if cands:
            return min(cands, key=lambda p: p.stat().st_size)
    return None


def check_verdict_json(body_text, kind):
    """Detect responses must be valid JSON with no NaN in p_fake_fused/trust/risk."""
    if "NaN" in body_text or "Infinity" in body_text:
        issues.append(f"{kind}: response body contains NaN/Infinity literal")
        return {}
    try:
        v = json.loads(body_text)
    except Exception as e:
        issues.append(f"{kind}: response is not valid JSON: {e}")
        return {}
    for k in ("p_fake_fused", "trust", "risk"):
        if k not in v:
            issues.append(f"{kind}: response missing {k}")
        elif isinstance(v[k], float) and v[k] != v[k]:
            issues.append(f"{kind}: {k} is NaN")
    return v


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        console, perrors = [], []
        page.on("console", lambda m: console.append((m.type, m.text)))
        page.on("pageerror", lambda e: perrors.append(str(e)))

        print("[1] load UI ...", flush=True)
        page.goto("http://127.0.0.1:8420/", wait_until="networkidle", timeout=60000)
        time.sleep(2)
        shot(page, "01_hero")

        print("[2] health chip:", page.locator("#healthChip").inner_text(), flush=True)
        if "ok" not in page.locator("#healthChip").inner_text():
            issues.append("health chip not ok")

        print("[2b] health API contract ...", flush=True)
        hr = page.request.get("http://127.0.0.1:8420/api/health")
        h = hr.json() if hr.ok else {}
        print("    stacker:", h.get("stacker"), "| n_modules:", h.get("n_modules"),
              flush=True)
        if not hr.ok:
            issues.append(f"health returned {hr.status}")
        if "stacker" not in h:
            issues.append("health response missing 'stacker'")
        if "n_modules" not in h:
            issues.append("health response missing 'n_modules'")

        if IMG is None:
            browser.close()
            report(console, perrors)
            return 0

        print("[3] drop file:", IMG.name, flush=True)
        import base64
        b64 = base64.b64encode(IMG.read_bytes()).decode()
        page.evaluate("""async (b64) => {
            const buf = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
            const dt = new DataTransfer();
            dt.items.add(new File([buf], 'exam.jpg', {type:'image/jpeg'}));
            const dz = document.getElementById('dz');
            dz.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true}));
        }""", b64)
        time.sleep(1)
        go_disabled = page.locator("#go").is_disabled()
        print("    go disabled after drop:", go_disabled, flush=True)
        if go_disabled:
            issues.append("drop did not enable the button (setFile failed)")
            shot(page, "02_after_drop_failed")
            browser.close()
            report(console, perrors)
            return 1
        shot(page, "02_after_drop")

        print("[4] convene the court ...", flush=True)
        with page.expect_response("**/api/detect", timeout=180000) as resp:
            page.locator("#go").click()
        r = resp.value
        print("    /api/detect status:", r.status, flush=True)
        if r.status != 200:
            issues.append(f"detect returned {r.status}: " + r.text()[:300])
        else:
            iv = check_verdict_json(r.text(), "image")
            print("    image kind:", iv.get("kind"), "| p_fake_fused:",
                  iv.get("p_fake_fused"), flush=True)
        time.sleep(3)
        shot(page, "03_verdict")

        vtag = page.locator("#vtag").inner_text()
        vnum = page.locator("#vnum").inner_text()
        ncards = page.locator(".ecard").count()
        print(f"[5] verdict: {vtag} | p_fake={vnum} | evidence cards={ncards}", flush=True)
        if "DELIBERATING" in vtag:
            issues.append("verdict never rendered (still deliberating)")
        if ncards == 0:
            issues.append("no evidence cards rendered")

        shot(page, "04_evidence_full")

        vid = find_video_fixture()
        if vid is None:
            print("[6] SKIP video e2e: no .mp4 fixture found under artifacts/ or "
                  "data/ (not fabricating one)", flush=True)
        else:
            print("[6] drop video:", vid.relative_to(WS), flush=True)
            vb64 = base64.b64encode(vid.read_bytes()).decode()
            page.evaluate("""async (b64) => {
                const buf = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
                const dt = new DataTransfer();
                dt.items.add(new File([buf], 'fixture.mp4', {type:'video/mp4'}));
                const dz = document.getElementById('dz');
                dz.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true}));
            }""", vb64)
            time.sleep(1)
            if page.locator("#go").is_disabled():
                issues.append("video drop did not enable the button (setFile failed)")
            else:
                with page.expect_response("**/api/detect", timeout=180000) as vresp:
                    page.locator("#go").click()
                vr = vresp.value
                print("    /api/detect (video) status:", vr.status, flush=True)
                if vr.status != 200:
                    issues.append(f"video detect returned {vr.status}: " + vr.text()[:300])
                else:
                    vv = check_verdict_json(vr.text(), "video")
                    print(f"    video kind={vv.get('kind')} p_fake_fused="
                          f"{vv.get('p_fake_fused')} abstained={vv.get('abstained')}",
                          flush=True)
                    if vv.get("kind") != "video":
                        issues.append(f"video detect kind={vv.get('kind')!r} "
                                      "(expected 'video')")
                time.sleep(2)
                shot(page, "05_video_verdict")

        browser.close()

    report(console, perrors)
    return 1 if issues else 0


def report(console, perrors):
    errs = [t for ty, t in console if ty == "error"] + perrors
    print("\n=== console errors ===")
    for e in errs[:10]:
        print("  !", e[:220])
    if perrors:
        issues.append("page errors: " + "; ".join(perrors[:3]))
    if errs:
        issues.append(f"{len(errs)} console error(s)")
    print("\n=== ISSUES ===" if issues else "\n=== E2E PASS: user flow works, no errors ===")
    for i in issues:
        print("  *", i)
    (OUT / "report.json").write_text(json.dumps(
        {"issues": issues, "console": errs[:20], "screenshots": shots}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
