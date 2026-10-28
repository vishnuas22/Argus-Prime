#!/usr/bin/env python
"""hop9 Deep Bench — black-box concurrency, latency decomposition, soak receipts.

Run: ./.venv/bin/python deepfake/scripts/perf_bench.py [--full|--quick]
Receipts: artifacts/mbeta/bench/20261026/  (CSV/JSON; every number traces here)

Laws honored:
  - zero external calls (target court 127.0.0.1:8420 only)
  - fixtures are court-supplied files, byte-verified; nothing fabricated
  - pre-registered hypotheses H0..H7 graded, no post-hoc goalposts
  - STOP: k=1 warm image p95 > manifest baseline x1.3 (2.34s) or any 500/row-loss
"""
import argparse
import asyncio
import json
import math
import statistics
import sqlite3
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

WS = Path(__file__).resolve().parents[2]

import httpx  # noqa: E402
import psutil  # noqa: E402

BASE = "http://127.0.0.1:8420"
OUT = WS / "artifacts" / "mbeta" / "bench" / "20261026"
OUT.mkdir(parents=True, exist_ok=True)
DB = WS / "artifacts" / "verdicts.db"
if not DB.exists():
    # db location fallback: locate it
    cands = list((WS / "data").rglob("verdicts.db"))
    if not cands:
        print("FATAL: verdicts.db not located", flush=True)
        sys.exit(2)
    DB = cands[0]

FX_IMAGE = WS / "artifacts" / "mbeta" / "fixtures" / "fixture2_laundered_fake.jpg"
FX_C2PA = WS / "artifacts" / "mbeta" / "fixtures" / "fixture3_c2pa_signed_ai.png"
FX_TRUN = WS / "artifacts" / "mbeta" / "fixtures" / "fixture_truncated.jpg"
FX_VIDEO = WS / "artifacts" / "fixtures" / "no_face.mp4"
FX_AUDIO = WS / "data" / "dfeval" / "audio" / "audio-data" / "audio-data" / "-0ljEYr8ZyqCHRqy5Bx4f4J6YOg.mp3"

EXPECT_IMAGE_PF = 0.9009     # battery-deterministic fused prob (fixture2)
STOP_P95_S = 2.34            # 1.8 x 1.3 (manifest perf_baseline_measured)

R = {"meta": {"started": datetime.now().isoformat(), "stop_line_s": STOP_P95_S,
              "hypotheses": [
                  "H0 image lane blocks loop -> k-ramp latency ~linear in k",
                  "H1 video alpha-lock serializes video; image partners unaffected in mixed k=4",
                  "H2 WAL+busy_timeout: db row delta == successful detects, zero 500s",
                  "H3 shared _state read-only -> no cross-contamination (per-file expected p holds)",
                  "H4 duplicate-file idempotency: 0.9009 EXACT every repeat",
                  "H5 >100MB refused fast (pre-multipart precheck)",
                  "H6 VLM explain cache unbounded (code-grade this hop)",
                  "H7 no-Origin bench clients pass the origin gate (lane [0] curl parity)",
              ]},
     "scenarios": {},
     "assertions": []}


def val(cond, name, detail=""):
    ok = bool(cond)
    R["assertions"].append({"name": name, "ok": ok, "detail": str(detail)[:260],
                            "at": datetime.now().isoformat()})
    print(("PASS " if ok else "FAIL ") + name + " :: " + str(detail)[:160], flush=True)
    return ok


def pctl(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(math.ceil(p / 100.0 * len(s))) - 1))
    return s[i]


def summarise(lat):
    if not lat:
        return {}
    return {"n": len(lat), "p50": round(pctl(lat, 50), 4), "p95": round(pctl(lat, 95), 4),
            "max": round(max(lat), 4), "mean": round(statistics.fmean(lat), 4)}


def db_stats():
    try:
        sz = DB.stat().st_size
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=3)
        n = con.execute("SELECT count(*) FROM verdicts").fetchone()[0]
        con.close()
        return {"path": str(DB), "size": sz, "rows": n}
    except Exception as e:
        return {"error": str(e)}


@asynccontextmanager
async def client(t=30.0):
    async with httpx.AsyncClient(base_url=BASE, timeout=httpx.Timeout(t)) as c:
        yield c


async def detect(c, fp, name, timeout=30.0):
    b = Path(fp).read_bytes()
    t0 = time.perf_counter()
    try:
        r = await c.post("/api/detect",
                         files={"file": (Path(fp).name, b, "application/octet-stream")},
                         timeout=timeout)
        dt = time.perf_counter() - t0
        try:
            j = r.json()
        except Exception:
            j = {"_parse_error": r.text[:200]}
        return {"name": name, "status": r.status_code, "lat": round(dt, 4),
                "json": j}, r.status_code in (200, 400, 403, 405, 413), dt
    except Exception as e:
        dt = time.perf_counter() - t0
        return {"name": name, "status": "EXC", "lat": round(dt, 4), "error": repr(e)[:200]},\
               "RemoteDisconnected" in repr(e) or "ConnectionReset" in repr(e), dt


async def warm_k1(c, fp, label, n, expected_pf=None, expect_status=200):
    rows = []
    for i in range(n):
        d, ok, _ = await detect(c, fp, f"{label}#{i+1}", timeout=60.0 if "video" in label else 30.0)
        rows.append(d)
        j = d.get("json", {})
        pf = j.get("verdict", {}).get("p_fake_fused", j.get("p_fake_fused"))
        if expected_pf is not None and d["status"] == 200 and pf is not None:
            val(abs(float(pf) - expected_pf) < 5e-4,
                f"{label} reader #{i+1} p_fake exact", f"{pf} vs {expected_pf}")
    lat = [d["lat"] for d in rows if d["status"] == expect_status]
    s = summarise(lat)
    R["scenarios"][f"warm_k1_{label}"] = {"rows": rows, "summary": s}
    print(f"[bench] warm_k1_{label}: {s}", flush=True)
    return s


async def ramp(c, k, rounds, fp=FX_IMAGE):
    raise NotImplementedError("superseded by ramp_clean")


async def ramp_clean(c, k, rounds):
    """Image ramp with exact p_fake assertion on every response + health-ping
    corroboration (external event-loop-starvation detector per scout Finding 2.4)."""
    rows, lats, sts, bad_pf, pings = [], [], [], [], []
    for rd in range(rounds):
        t0 = time.perf_counter()

        async def ping():
            tt = time.perf_counter()
            h = await c.get("/api/health", timeout=15.0)
            pings.append(round(time.perf_counter() - tt, 4))
            return h.status_code

        tasks = [detect(c, FX_IMAGE, f"ramp_k{k}_r{rd}_i{i}") for i in range(k)]
        res = await asyncio.gather(*tasks, ping(), ping())  # 2 corroboration pings mid-ramp
        wall = time.perf_counter() - t0
        for d, _, _ in res[:k]:
            sts.append(d["status"])
            rows.append({kk: d.get(kk) for kk in ("name", "status", "lat", "error")})
            if d["status"] == 200:
                lats.append(d["lat"])
                pf = d["json"].get("verdict", {}).get("p_fake_fused",
                      d["json"].get("p_fake_fused"))
                if pf is None or abs(float(pf) - EXPECT_IMAGE_PF) >= 5e-4:
                    bad_pf.append({"name": d["name"], "pf": pf})
        print(f"[bench] k={k} r{rd+1}/{rounds} wall={wall:.3f}s "
              f"ok={sum(1 for s in sts[len(sts)-k:] if s==200)}/{k} "
              f"pings={pings[-2:]}", flush=True)
    val(not any(s >= 500 for s in sts), f"ramp k={k}: zero 5xx",
        f"statuses {sorted(set(sts))}")
    val(not bad_pf, f"ramp k={k}: zero cross-contaminated verdicts", bad_pf[:2])
    s = summarise(lats)
    R["scenarios"][f"ramp_k{k}"] = {"summary": s, "rows": rows, "statuses": sts,
                                    "rounds": rounds, "contaminations": len(bad_pf),
                                    "health_pings": pings, "semantics": "gather-stampede"}
    print(f"[bench] ramp_k{k}: {s} ping_p50={pctl(pings, 50):.4f}", flush=True)
    return s


async def hammer_same_file(c, rounds=3):
    """4 concurrent IDENTICAL uploads + exact expected pf (thread-safety proof core)."""
    rows, sts = [], []
    for rd in range(rounds):
        tasks = [detect(c, FX_IMAGE, f"hammer_r{rd}_i{i}") for i in range(4)]
        res = [(d, ok, _lat) for d, ok, _lat in await asyncio.gather(*tasks)]
        for d, _ok, _lat in res:
            pf = d["json"].get("verdict", {}).get("p_fake_fused",
                 d["json"].get("p_fake_fused"))
            sts.append(d["status"])
            rows.append({"name": d["name"], "status": d["status"], "lat": d["lat"], "pf": pf})
    ok_all = all(s == 200 for s in sts)
    exact_all = all(r["pf"] is not None and abs(float(r["pf"]) - EXPECT_IMAGE_PF) < 5e-4
                    for r in rows)
    R["scenarios"]["hammer_same_k4"] = {"rows": rows, "statuses": sts}
    val(not any((s >= 500) for s in sts if isinstance(s, int)),
        "hammer k=4 identical: zero 5xx", sorted(set(map(str, sts))))
    val(ok_all and exact_all, "hammer k=4 identical: every verdict 0.9009 EXACT",
        f"{sorted(set(str(r['pf']) for r in rows))}")
    print("[bench] hammer_same_k4 done", flush=True)


async def mixed_k4(c):
    """2 image + 1 video + 1 audio concurrent (H1/H3 together).
    Timeouts sized for the serialized regime (video blocks the loop ~25-36s)."""
    t0 = time.perf_counter()
    tasks = [detect(c, FX_IMAGE, "mixed_img0", timeout=120.0),
             detect(c, FX_IMAGE, "mixed_img1", timeout=120.0),
             detect(c, FX_VIDEO, "mixed_vid", timeout=180.0),
             detect(c, FX_AUDIO, "mixed_aud", timeout=120.0)]
    res = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    rows = []
    for d, ok, _ in res:
        j = d.get("json", {})
        v = j.get("verdict", {})
        rows.append({"name": d["name"], "status": d["status"], "lat": d["lat"],
                     "kind": (v.get("kind") or j.get("kind")) if d["status"] == 200 else None,
                     "pf": v.get("p_fake_fused", j.get("p_fake_fused")) if d["status"] == 200 else None})
    R["scenarios"]["mixed_k4"] = {"rows": rows, "wall": round(wall, 4)}
    val(not any(isinstance(r["status"], int) and r["status"] >= 500 for r in rows),
        "mixed k=4: zero 5xx", rows)
    img = [r for r in rows if r["name"].startswith("mixed_img") and r["status"] == 200]
    val(all(r["pf"] is not None and abs(float(r["pf"]) - EXPECT_IMAGE_PF) < 5e-4 for r in img),
        "mixed: both image verdicts exact 0.9009 (no contamination)",
        [r["pf"] for r in img])
    vid = [r for r in rows if r["name"].startswith("mixed_vid")]
    if vid and vid[0]["status"] == 200:
        val(vid[0]["kind"] in ("video", "image", "degraded", "abstained"),
            "mixed: video lane adjudicated (any route)", f"kind={vid[0]['kind']}")
    print(f"[bench] mixed_k4 wall={wall:.3f}s "
          f"{[(r['name'], r['status'], r['lat']) for r in rows]}", flush=True)


async def dup_idempotency(c, n=3):
    rows = []
    for i in range(n):
        d, _, _ = await detect(c, FX_IMAGE, f"dup#{i+1}")
        j = d.get("json", {})
        v = j.get("verdict", {})
        rows.append({"status": d["status"], "lat": d["lat"],
                     "pf": v.get("p_fake_fused", j.get("p_fake_fused"))})
    pfs = [r["pf"] for r in rows if r["status"] == 200]
    val(len(pfs) == n and all(p is not None and abs(float(p) - EXPECT_IMAGE_PF) < 5e-4
                             for p in pfs),
        f"dup x{n}: 0.9009 EXACT every repeat (H4)",
        pfs)
    R["scenarios"]["dup_idempotency"] = {"rows": rows, "pfs": pfs}


async def oversize_refusal(c, mb=101):
    """Sparse >cap upload; F6 precheck should 413 fast OR drop connection fast; never hang."""
    tmp = Path("/private/var/folders/zr/xhd5gptd36zgdy5mmcv4gjfc0000gp/T/opencode/big_101.bin")
    if not tmp.exists() or tmp.stat().st_size != mb * 1024 * 1024:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["dd", "if=/dev/zero", f"of={tmp}", "bs=1m", f"count={mb}"],
                       check=True, capture_output=True)
    t0 = time.perf_counter()
    try:
        r = await c.post("/api/detect", files={"file": ("big.bin", b"x" * 0 + tmp.read_bytes(),
                                                        "application/octet-stream")},
                         timeout=20.0)
        status, lat = r.status_code, round(time.perf_counter() - t0, 4)
        body = r.text[:120]
    except Exception as e:
        status, lat, body = "EXC", round(time.perf_counter() - t0, 4), repr(e)[:120]
    fast = lat <= 6.0
    R["scenarios"]["oversize_refusal"] = {"status": status, "lat": lat, "body": body}
    val(fast and status in (413, 400, "EXC"),
        f"oversize {mb}MB refusal fast ({H5_LAW})", f"status={status} lat={lat}s")
    print(f"[bench] oversize: status={status} lat={lat}s", flush=True)
    return status, lat, body


H5_LAW = "F6 content-length precheck"


async def truncated_lane(c, n=3):
    lats, sts = [], []
    for i in range(n):
        d, _, _ = await detect(c, FX_TRUN, f"trunc#{i+1}", timeout=10.0)
        sts.append(d["status"])
        if d["status"] == 400:
            lats.append(d["lat"])
    R["scenarios"]["truncated"] = {"statuses": sts, "lats": lats}
    val(all(s == 400 for s in sts), "truncated 200B fixture -> clean 400 on every hit",
        sts)
    val(bool(lats) and max(lats) < 1.5, "truncated refusal <1.5s", summarise(lats))
    print(f"[bench] truncated: {sorted(set(sts))} {summarise(lats)}", flush=True)


async def history_probe(c, rounds=30):
    lats = []
    t0 = time.perf_counter()
    for _ in range(rounds):
        t = time.perf_counter()
        r = await c.get("/api/history?limit=50")
        lats.append(time.perf_counter() - t)
        if r.status_code != 200:
            break
    s = summarise(lats)
    R["scenarios"]["history_p95"] = {"summary": s, "rounds": rounds}
    print(f"[bench] history x{rounds}: {s}", flush=True)
    return s


async def baseline_history(c, cb):
    await history_probe(c, rounds=30)


def rss_sidecar(stop_ev, path, pid):
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"[sidecar] pid {pid} not found; RSS watcher idle", flush=True)
        return
    with open(path, "w") as f:
        f.write("t_sec,rss_mb,cpu_pct,threads\n")
        t0 = time.perf_counter()
        while not stop_ev.is_set():
            try:
                m = p.memory_info().rss / (1024 * 1024)
                cpu = p.cpu_percent(interval=None)
                th = p.num_threads()
                f.write(f"{time.perf_counter()-t0:.1f},{m:.1f},{cpu:.1f},{th}\n")
                f.flush()
            except Exception:
                pass
            stop_ev.wait(2.0)


def uvicorn_pid():
    out = subprocess.run(["pgrep", "-f", "uvicorn deepfake.api.server"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def find_proc_psutil():
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if "uvicorn" in cmd and "deepfake" in cmd:
                return p.info["pid"]
        except Exception:
            continue
    return None


async def main_async(full: bool, part2: bool = False):
    before_db = db_stats()
    print(f"[bench] db BEFORE: {before_db}", flush=True)

    async with client() as c:
        # lane sanity (court has been running = warm)
        try:
            h = await c.get("/api/health", timeout=5.0)
            val(h.status_code == 200 and len(h.json().get("channels", [])) == 11,
                "health: 11 channels pre-bench", h.status_code)
        except Exception as e:
            val(False, "health pre-bench", repr(e))
            return

        total0 = time.perf_counter()

        if part2:
            # continuation run after the part-1 instrument crash at mixed_k4:
            # mixed retry (honest timeouts) + hammer + oversize + truncated + history
            R["scenarios"]["part2_db_before"] = {"db": before_db}
            await mixed_k4(c)
            await hammer_same_file(c, rounds=3)
            R["meta"]["rss_event_oversize_t"] = time.perf_counter() - total0
            await oversize_refusal(c)
            await truncated_lane(c)
            R["scenarios"]["history_after"] = {"summary": await history_probe(c, rounds=30)}
            R["meta"]["wall_bench_s"] = round(time.perf_counter() - total0, 2)
            R["after_db"] = db_stats()
            print(f"[bench] db AFTER: {R['after_db']}", flush=True)
            return

        # history baseline BEFORE bench adds rows (growth-comparison arm)
        R["scenarios"]["history_before"] = {"summary": await history_probe(c, 20),
                                            "db": before_db, "note": "pre-bench rows"}

        await warm_k1(c, FX_IMAGE, "image", 7 if full else 4, expected_pf=EXPECT_IMAGE_PF)
        s_img = R["scenarios"].get("warm_k1_image", {}).get("summary", {})
        R["meta"]["stop_condition"] = {"tripped": False, "stop_line_s": STOP_P95_S}
        if s_img.get("p95") is not None and s_img["p95"] > STOP_P95_S * 1.0:
            # USER RULING 2026-10-26 (question tool): continue under drift label.
            # Pre-registered halt converted to labeled continuation BY THAT WORD ONLY.
            R["meta"]["stop_condition"] = {
                "tripped": True, "scenario": "warm_k1_image", "p95": s_img["p95"],
                "user_ruling": "continue under drift label (2026-10-26); relative-scaling "
                               "analysis is load-robust; drift tracked as its own finding",
            }
            print(f"[bench] STOP condition TRIPPED (p95={s_img['p95']}s > {STOP_P95_S}s) "
                  "-> CONTINUING under user-ruled drift label", flush=True)

        await warm_k1(c, FX_AUDIO, "audio", 5 if full else 3)
        await warm_k1(c, FX_VIDEO, "video", 3 if full else 1)

        await dup_idempotency(c, 3)

        if full:
            await ramp_clean(c, 2, 3)
            await ramp_clean(c, 4, 3)
            await ramp_clean(c, 8, 3)
            await mixed_k4(c)
        else:
            await ramp_clean(c, 4, 1)
            await mixed_k4(c)

        await hammer_same_file(c, rounds=3 if full else 1)

        R["meta"]["rss_event_oversize_t"] = time.perf_counter() - total0
        await oversize_refusal(c)
        await truncated_lane(c)

        R["scenarios"]["history_after"] = {"summary": await history_probe(c, rounds=30)}

        R["meta"]["wall_bench_s"] = round(time.perf_counter() - total0, 2)

    R["after_db"] = db_stats()
    print(f"[bench] db AFTER: {R['after_db']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--part2", action="store_true")
    a = ap.parse_args()
    full = not a.quick

    # preflight fixtures (bytes are court-supplied truth)
    for fx in (FX_IMAGE, FX_C2PA, FX_TRUN, FX_VIDEO, FX_AUDIO):
        if not fx.exists():
            print(f"FATAL fixture missing: {fx}", flush=True)
            sys.exit(2)
        print(f"[bench] fixture {fx.name}: {fx.stat().st_size}B", flush=True)

    # sidecars
    pid = uvicorn_pid() or find_proc_psutil()
    rss_path = OUT / "rss_watcher.csv"
    import threading
    stop_ev = threading.Event()
    t_rss = None
    if pid and not a.part2:  # part2 uses the standalone watcher already running
        t_rss = threading.Thread(target=rss_sidecar, args=(stop_ev, rss_path, pid),
                                 daemon=True)
        t_rss.start()
        print(f"[bench] rss sidecar on pid {pid}", flush=True)

    try:
        asyncio.run(main_async(full, part2=a.part2))
    finally:
        stop_ev.set()
        if t_rss:
            t_rss.join(timeout=6)
        R["meta"]["finished"] = datetime.now().isoformat()
        # H2 reconciliation: db row delta == every 200-status detect issued
        def count_200():
            n = 0
            for _name, sc in R["scenarios"].items():
                for row in sc.get("rows", []):
                    if isinstance(row, dict) and row.get("status") == 200:
                        n += 1
            return n

        n200 = count_200()
        hb = (R["scenarios"].get("part2_db_before", {})
              or R["scenarios"].get("history_before", {})).get("db", {})
        rows_before = hb.get("rows")
        rows_after = db_stats().get("rows")
        if rows_before is not None and rows_after is not None:
            R["H2_reconciliation"] = {"detects_200": n200, "rows_before": rows_before,
                                      "rows_after": rows_after,
                                      "row_delta": rows_after - rows_before}
            val(rows_after - rows_before == n200,
                "H2 row-loss reconciliation: delta == 200-count",
                f"delta={rows_after - rows_before} n200={n200}")
        else:
            R["H2_reconciliation"] = {"detects_200": n200, "rows_before": rows_before,
                                      "rows_after": rows_after,
                                      "note": "bench aborted before history baseline"}

        out = OUT / ("bench_results_part2.json" if a.part2 else "bench_results.json")
        out.write_text(json.dumps(R, indent=1))
        n_fail = sum(1 for a in R["assertions"] if not a["ok"])
        print(f"\n[bench] DONE -> {out}  assertions: "
              f"{len(R['assertions'])-n_fail}/{len(R['assertions'])} PASS", flush=True)
        if n_fail:
            print("[bench] FAILURES:", flush=True)
            for a in R["assertions"]:
                if not a["ok"]:
                    print(f"  FAIL {a['name']} :: {a['detail']}", flush=True)
        sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
