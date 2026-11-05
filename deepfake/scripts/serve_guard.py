#!/usr/bin/env python3
"""Court-server supervision (P1 ops fix, 2026-11-04).

The REG battery depends on the court at 127.0.0.1:8420, which until now ran
under bare nohup: a mid-battery death produced confusing test failures and no
recovery. This guard makes the server's liveness explicit and self-healing:

  --once   ensure the server is healthy; start it if absent; exit 0 healthy /
           2 unhealthy (usable as a pre-battery gate)
  --watch  daemon loop: poll /api/health, restart on death, log every action;
           give up (exit 2) after --max-restarts consecutive failed starts
  --stop   terminate the server cleanly and verify it is down

The server itself is NEVER modified here — supervision only observes process
state and (re)starts the exact documented command:
    .venv/bin/python -m uvicorn deepfake.api.server:app --port 8420
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import urllib.request

WS = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8420"
HEALTH = f"{BASE}/api/health"
LOG_DIR = WS / "logs"
GUARD_LOG = LOG_DIR / "serve_guard.log"
PGREP = "uvicorn deepfake.api.server"


def log(msg: str) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(GUARD_LOG, "a") as f:
        f.write(line + "\n")


def pid() -> int | None:
    r = subprocess.run(["pgrep", "-f", PGREP], capture_output=True, text=True)
    out = r.stdout.split()
    return int(out[0]) if out else None


def healthy(timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(HEALTH, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def start() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    logf = open(LOG_DIR / "api.log", "ab")
    p = subprocess.Popen(
        [str(WS / ".venv" / "bin" / "python"), "-m", "uvicorn",
         "deepfake.api.server:app", "--port", "8420"],
        cwd=str(WS), stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True)
    log(f"started uvicorn pid {p.pid} (logs/api.log)")
    return p.pid


def await_health(grace_s: int = 150) -> bool:
    """Poll until healthy or grace expires (model load is heavy on cold start)."""
    t0 = time.time()
    while time.time() - t0 < grace_s:
        if healthy():
            return True
        time.sleep(2.0)
    return False


def stop() -> int:
    p = pid()
    if p is None:
        print("server not running")
        return 0
    subprocess.run(["pkill", "-f", PGREP], capture_output=True)
    for _ in range(30):
        if pid() is None:
            log(f"stopped pid {p}")
            print("server stopped")
            return 0
        time.sleep(1.0)
    subprocess.run(["pkill", "-9", "-f", PGREP], capture_output=True)
    time.sleep(2.0)
    if pid() is None:
        log(f"force-stopped pid {p}")
        print("server force-stopped")
        return 0
    print("FATAL: server did not stop")
    return 2


def once() -> int:
    if healthy():
        print("healthy (already running)")
        return 0
    p = pid()
    if p is not None:
        log(f"process {p} present but unhealthy — killing stale instance")
        subprocess.run(["pkill", "-f", PGREP], capture_output=True)
        time.sleep(3.0)
    start()
    if await_health():
        log("health OK after start")
        print("healthy (started)")
        return 0
    log("FATAL: start did not become healthy within grace")
    print("FATAL: server unhealthy after start (see logs/api.log)")
    return 2


def watch(interval_s: float, max_restarts: int) -> int:
    log(f"watch mode: interval {interval_s}s, max consecutive restarts "
        f"{max_restarts}")
    consecutive = 0
    while True:
        if healthy():
            consecutive = 0
            time.sleep(interval_s)
            continue
        p = pid()
        if p is not None:
            log(f"unhealthy with pid {p} — killing stale instance")
            subprocess.run(["pkill", "-f", PGREP], capture_output=True)
            time.sleep(3.0)
        consecutive += 1
        if consecutive > max_restarts:
            log(f"FATAL: {max_restarts} consecutive failed starts — giving up")
            return 2
        start()
        if await_health():
            log("recovered after restart")
            consecutive = 0
        else:
            log("restart did not become healthy")
        time.sleep(interval_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true",
                   help="ensure healthy now; exit 0/2")
    g.add_argument("--watch", action="store_true",
                   help="daemon: poll + restart on death")
    g.add_argument("--stop", action="store_true",
                   help="stop the server and verify")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--max-restarts", type=int, default=3)
    a = ap.parse_args()
    if a.stop:
        return stop()
    if a.once:
        return once()
    return watch(a.interval, a.max_restarts)


if __name__ == "__main__":
    sys.exit(main())
