#!/usr/bin/env python3
"""Read-only Jetson host telemetry; Ctrl+C stops only this collector's child."""
import argparse
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time


def query(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return {"command": cmd, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": cmd, "error": str(exc)}


def write_line(f, **data):
    f.write(json.dumps({"timestamp": time.time(), **data}, ensure_ascii=False) + "\n")
    f.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="New directory")
    p.add_argument("--interval-ms", type=int, default=1000)
    p.add_argument("--duration", type=int, default=0, help="Seconds; 0 until Ctrl+C")
    args = p.parse_args()
    if args.interval_ms < 100 or args.duration < 0:
        p.error("interval must be >=100 ms; duration must be >=0")
    tegra = shutil.which("tegrastats") or next((str(x) for x in (Path('/usr/bin/tegrastats'), Path('/usr/sbin/tegrastats'), Path('/home/nvidia/tegrastats')) if x.is_file()), None)
    if not tegra:
        p.error("tegrastats missing. Run this collector on the Jetson HOST, not inside the inference container.")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    meta = {"start_timestamp": time.time(), "uname": list(os.uname()), "arguments": vars(args),
            "queries": [query(["nvpmodel", "-q", "--verbose"]), query(["jetson_clocks", "--show"])],
            "timestamp_note": "Unix wall time when each stdout line is received, not an exact device sampling timestamp. stdbuf -oL is used when present."}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    cmd = [tegra, "--interval", str(args.interval_ms)]
    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL", *cmd]
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    start = time.monotonic()
    count = 0
    with (out / "tegrastats_stderr.log").open("wb") as err, (out / "tegrastats.jsonl").open("x", encoding="utf-8") as telemetry, (out / "processes.jsonl").open("x", encoding="utf-8") as processes:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        pending = b""
        next_ps = 0
        print(f"Recording Jetson telemetry to {out}. Keep this terminal open; Ctrl+C after ABA completes.", flush=True)
        try:
            while not stop and (not args.duration or time.monotonic() - start < args.duration):
                for key, _ in selector.select(timeout=.5):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        raise RuntimeError("tegrastats exited; inspect tegrastats_stderr.log")
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        raw = line.decode('utf-8', errors='replace')
                        valid_sample = 'RAM ' in raw and 'CPU [' in raw
                        write_line(telemetry, raw=raw, recognized_sample=valid_sample)
                        if valid_sample:
                            count += 1
                            if count == 1:
                                print("First tegrastats sample captured; you can start run_aba.py now.", flush=True)
                if time.monotonic() >= next_ps:
                    # Lifetime-average CPU use plus /proc/stat; not GPU per-process accounting.
                    snap = query(["ps", "-eo", "pid,ppid,stat,pcpu,pmem,rss,comm", "--sort=-pcpu"])
                    write_line(processes, ps=snap, proc_stat=Path('/proc/stat').read_text(),
                               loadavg=Path('/proc/loadavg').read_text())
                    next_ps = time.monotonic() + 5
                if count == 0 and time.monotonic() - start > 15:
                    raise RuntimeError("No tegrastats samples after 15 seconds; inspect stderr, privileges and stdout buffering")
        finally:
            selector.close()
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
            proc.stdout.close()
            (out / "stopped.json").write_text(json.dumps({"timestamp": time.time(), "samples": count, "tegrastats_returncode": proc.returncode}, indent=2) + "\n", encoding='utf-8')
    print(f"Saved {count} samples to {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
