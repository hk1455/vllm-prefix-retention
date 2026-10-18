#!/usr/bin/env python3
"""Short LRU -> SLRU -> LRU diagnostic. Run inside the existing vLLM environment."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def event(out, kind, **extra):
    row = {"timestamp": time.time(), "event": kind, **extra}
    with (out / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] {kind} {extra}", flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_free_port(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is occupied. Stop your existing server in its own terminal first; no existing process was killed.") from exc


def group_alive(proc):
    proc.poll()  # reap the leader, but also check children left in its group
    try:
        os.killpg(proc.pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_owned(proc, grace=60):
    # Every managed child was started with start_new_session=True.
    # Never pkill/killall or stop a pre-existing server.
    if not group_alive(proc):
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while group_alive(proc) and time.monotonic() < deadline:
        time.sleep(0.2)
    if group_alive(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=10)


def verify_log(text, policy, blocks):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    policies = re.findall(r"PREFIX_RETENTION policy=(\w+)\b", text)
    sizes = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    if policies != [policy] or not sizes or any(int(x.replace(',', '')) != blocks * 16 for x in sizes):
        raise RuntimeError("Actual policy/capacity not confirmed in server.log; refusing to benchmark labels only.")
    return {"policy": policies[0], "kv_tokens": int(sizes[-1].replace(',', ''))}


def headers():
    return {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]} if os.environ.get("VLLM_API_KEY") else {}


def wait_ready(proc, url, logfile, policy, blocks, timeout):
    deadline = time.monotonic() + timeout
    last_notice = 0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited ({proc.returncode}); read {logfile}")
        try:
            with urllib.request.urlopen(urllib.request.Request(url + "/health", headers=headers()), timeout=2) as response:
                healthy = response.status == 200
        except (OSError, ValueError):
            healthy = False
        if healthy:
            return verify_log(logfile.read_text(encoding="utf-8", errors="replace"), policy, blocks)
        if time.monotonic() - last_notice > 30:
            print(f"Waiting for server; log: {logfile}", flush=True)
            last_notice = time.monotonic()
        time.sleep(1)
    raise RuntimeError(f"Server startup timed out after {timeout}s")


def process_snapshot(path):
    p = subprocess.run(["ps", "-eo", "pid,ppid,stat,pcpu,pmem,rss,comm", "--sort=-pcpu"],
                       capture_output=True, text=True, timeout=10)
    save(path, {"timestamp": time.time(), "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr})


def validate_cold(row, expected_tokens):
    if row["cached_tokens"] != 0:
        raise RuntimeError("Reset failed: cold request had cached tokens")
    if row["prompt_tokens"] != expected_tokens or row["completion_tokens"] != 1:
        raise RuntimeError("Cold request input/output token count mismatch")


async def cold_probe(bench, args, out):
    import aiohttp
    out.mkdir()
    cfg = SimpleNamespace(base_url=args.base_url, model="qwen3-4b", output_tokens=1,
                          prefix_tokens=3072, pool_seed=20261013, block_size=16, metrics_timeout=60)
    timeout = aiohttp.ClientTimeout(total=180, connect=15)
    async with aiohttp.ClientSession(headers=headers(), timeout=timeout) as session:
        client = bench.Client(session, cfg)
        version = await client.json("GET", "/version")
        if not str(version.get("version", "")).startswith("0.9.2"):
            raise RuntimeError(f"Expected vLLM 0.9.2; got {version}")
        # Validates positive cached_tokens exposure and cold/hit/reset behavior.
        await bench.preflight(client, cfg, out)
        text = await bench.make_prefix(client, 3072, "diagnostic-cold-fixed-v1")
        text += "\nReply with OK."
        ids = await client.chat_tokens(text)
        digest = bench.digest(ids)
        save(out / "input.json", {"text": text, "token_ids_sha256": digest, "prompt_tokens": len(ids)})
        rows = []
        with (out / "requests.jsonl").open("x", encoding="utf-8") as f:
            for i in range(args.cold_warmup + args.cold_requests):
                reset_start = time.time()
                await client.reset()
                reset_end = time.time()
                dispatched = time.time()
                r = await client.completion(text, f"aba-cold-{i}")
                row = {"index": i, "phase": "warmup" if i < args.cold_warmup else "measure",
                       "dispatch_time": dispatched, "finish_time": time.time(),
                       "reset_start": reset_start, "reset_end": reset_end,
                       "token_ids_sha256": digest, **r,
                       "cached_tokens": r["cached_tokens_raw"] or 0}
                # Missing zero values are accepted only after the positive-hit preflight.
                f.write(json.dumps(row) + "\n")
                f.flush()
                validate_cold(row, len(ids))
                rows.append(row)
                print(f"cold {i+1}/{args.cold_warmup + args.cold_requests}: {r['ttft_ms']:.1f} ms", flush=True)
        await client.reset()
    measured = [r["ttft_ms"] for r in rows if r["phase"] == "measure"]
    result = {"status": "complete", "token_ids_sha256": digest, "requests": len(measured),
              "prompt_tokens": len(ids), "ttft_ms_mean": statistics.mean(measured),
              "ttft_ms_p50": statistics.median(measured), "ttft_ms_p95": bench.percentile(measured, .95),
              "all_cached_tokens_zero": True, "reset_time_excluded": True}
    save(out / "summary.json", result)
    return result


def run_logged(cmd, cwd, env, logfile):
    with logfile.open("wb") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                try:
                    code = proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f"Zipf running; progress: {logfile}", flush=True)
            if code:
                raise RuntimeError(f"Command failed ({code}); read {logfile}")
        finally:
            stop_owned(proc, grace=15)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/tmp/vllm_prefix_retention", help="Existing installed benchmark bundle")
    p.add_argument("--out", required=True, help="New directory; never overwrite")
    p.add_argument("--blocks", type=int, default=2048)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cold-requests", type=int, default=30)
    p.add_argument("--cold-warmup", type=int, default=5)
    p.add_argument("--zipf-warmup", type=int, default=50)
    p.add_argument("--zipf-requests", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--startup-timeout", type=int, default=900)
    args = p.parse_args()
    if sys.platform != "linux":
        p.error("Run this orchestrator on the Jetson/Linux vLLM environment")
    if min(args.blocks, args.cold_requests, args.zipf_requests, args.startup_timeout) <= 0 or min(args.cold_warmup, args.zipf_warmup) < 0 or not 1 <= args.port <= 65535:
        p.error("Invalid counts, port, or timeout")
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    names = ("start_server.sh", "manage_patch.py", "zipf_prefix_bench.py", "compare_results.py")
    for name in names:
        if not (root / name).is_file():
            p.error(f"Missing {root / name}")
    require_free_port(args.port)
    bench = load_module("diagnostic_bench", root / "zipf_prefix_bench.py")
    compare = load_module("diagnostic_compare", root / "compare_results.py")
    out.mkdir(parents=True, exist_ok=False)
    args.base_url = f"http://127.0.0.1:{args.port}"
    env = {**os.environ, "PYTHON_BIN": sys.executable, "PYTHONUNBUFFERED": "1",
           "HOST": "127.0.0.1", "PORT": str(args.port), "VLLM_PREFIX_RETENTION_PROTECTED_RATIO": "0.5"}
    save(out / "manifest.json", {"arguments": vars(args), "timestamp": time.time(),
         "python": sys.version, "sequence": ["lru", "slru", "lru"],
         "environment": {k: env.get(k) for k in ("MODEL", "GPU_MEMORY_UTILIZATION", "CUDA_VISIBLE_DEVICES", "HOST", "PORT", "VLLM_PREFIX_RETENTION_PROTECTED_RATIO")},
         "source_sha256": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}})
    results = []
    try:
        for index, policy in enumerate(("lru", "slru", "lru"), 1):
            stage = out / f"{index:02d}_{policy}"
            stage.mkdir()
            require_free_port(args.port)
            process_snapshot(stage / "processes_before.json")
            event(out, "server_start", stage=stage.name, policy=policy)
            with (stage / "server.log").open("wb") as log:
                proc = subprocess.Popen(["bash", str(root / "start_server.sh"), policy, str(args.blocks)],
                          cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    actual = wait_ready(proc, args.base_url, stage / "server.log", policy, args.blocks, args.startup_timeout)
                    save(stage / "server_verified.json", actual)
                    event(out, "cold_start", stage=stage.name)
                    cold = asyncio.run(cold_probe(bench, args, stage / "cold"))
                    event(out, "cold_end", stage=stage.name)
                    cmd = [sys.executable, str(root / "zipf_prefix_bench.py"), "--base-url", args.base_url,
                           "--model", "qwen3-4b", "--prefixes", "100", "--prefix-tokens", "3072",
                           "--alpha", "1.0", "--seed", str(args.seed), "--pool-seed", "20261013",
                           "--warmup", str(args.zipf_warmup), "--requests", str(args.zipf_requests),
                           "--concurrency", "1", "--output-tokens", "1", "--gpu-blocks", str(args.blocks),
                           "--out", str(stage / "zipf")]
                    save(stage / "zipf_command.json", cmd)
                    event(out, "zipf_start", stage=stage.name)
                    run_logged(cmd, root, env, stage / "zipf_console.log")
                    zipf = json.loads((stage / "zipf/summary.json").read_text(encoding="utf-8"))
                    if results:
                        if cold["token_ids_sha256"] != results[0]["cold"]["token_ids_sha256"]:
                            raise RuntimeError("Cold prompt hash changed across stages")
                        compare.compare(results[0]["zipf"], zipf)
                    if proc.poll() is not None:
                        raise RuntimeError("Server exited during measurements")
                    results.append({"stage": stage.name, "cold": cold, "zipf": zipf})
                    event(out, "zipf_end", stage=stage.name)
                finally:
                    event(out, "server_stop", stage=stage.name)
                    stop_owned(proc)
            process_snapshot(stage / "processes_after.json")
            time.sleep(5)
        comparisons = {}
        for label, i, j in (("slru_vs_first_lru", 0, 1), ("last_lru_vs_first_lru", 0, 2), ("slru_vs_last_lru", 2, 1)):
            comparisons[label] = {"cold_p50_change_percent": 100 * (results[j]["cold"]["ttft_ms_p50"] / results[i]["cold"]["ttft_ms_p50"] - 1),
                                  "zipf_metrics": compare.compare(results[i]["zipf"], results[j]["zipf"])}
        save(out / "comparison.json", {"status": "complete", "runs": results, "comparisons": comparisons,
             "interpretation": "Short diagnostic, not a definitive performance benchmark. Pair hardware telemetry by Unix timestamps; no throttling cause inferred automatically."})
        lines = ["# ABA diagnostic", "", "| Stage | Cold TTFT p50 ms | Zipf mean TTFT ms | Repeat recompute tokens |", "|---|---:|---:|---:|"]
        for r in results:
            m = r["zipf"]["measurement"]
            lines.append(f"| {r['stage']} | {r['cold']['ttft_ms_p50']:.1f} | {m['ttft_ms_mean']:.1f} | {m['repeat_prefix_recompute_tokens']} |")
        lines += ["", "See comparison.json for warnings and comparisons. Hardware telemetry is required to investigate frequency changes; these latencies alone cannot establish throttling."]
        (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        event(out, "complete", output=str(out))
    except BaseException as exc:
        save(out / "failure.json", {"type": type(exc).__name__, "error": str(exc), "timestamp": time.time(), "completed_stages": len(results)})
        raise


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; owned child groups stopped, partial results preserved.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
