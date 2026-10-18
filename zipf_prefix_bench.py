#!/usr/bin/env python3
"""Zipf prefix-retention benchmark for vLLM 0.9.2 V1. Dependency: aiohttp.

Run against an otherwise idle, dedicated server. Uses the server tokenizer,
Chat API cached_tokens, and Prometheus counters; does not modify vLLM.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone

import aiohttp


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def lcp(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def percentile(values, q):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def ratio(a, b):
    return a / b if b else None


def parse_metrics(raw, model):
    families = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r'^([^\s{]+)(\{.*\})?\s+(\S+)', line)
        if not match:
            continue
        name, labels, value = match.groups()
        label = re.search(r'(?:^|[,\{])model_name="((?:\\.|[^"\\])*)"', labels or "")
        if label and json.loads('"' + label.group(1) + '"') != model:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            families.setdefault(name, []).append(number)
    names = {
        "queries": ["vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"],
        "hits": ["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"],
        "preemptions": ["vllm:num_preemptions_total"],
        "running": ["vllm:num_requests_running"],
        "waiting": ["vllm:num_requests_waiting"],
        "usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
        "success": ["vllm:request_success_total"],
        "queue_seconds_sum": ["vllm:request_queue_time_seconds_sum"],
        "queue_seconds_count": ["vllm:request_queue_time_seconds_count"],
    }
    result = {"timestamp": time.time(), "metric_names": {}}
    for key, candidates in names.items():
        name = next((n for n in candidates if n in families), None)
        result[key] = (max(families[name]) if key == "usage" else sum(families[name])) if name else None
        result["metric_names"][key] = name
    return result


class HTTPError(RuntimeError):
    def __init__(self, route, status, raw):
        self.status = status
        super().__init__(f"{route}: HTTP {status}: {raw[:1200]}")


class Client:
    def __init__(self, session, args):
        self.session, self.args = session, args
        self.base = args.base_url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]

    async def json(self, method, route, body=None):
        async with self.session.request(method, self.base + route, json=body) as response:
            raw = await response.text()
            if response.status != 200:
                raise HTTPError(route, response.status, raw)
            return json.loads(raw) if raw.strip() else None

    async def tokens(self, **kwargs):
        result = await self.json("POST", "/tokenize", {"model": self.args.model, **kwargs})
        return result["tokens"]

    def chat(self, text):
        return {"model": self.args.model, "messages": [{"role": "user", "content": text}],
                "add_special_tokens": False, "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False}}

    async def chat_tokens(self, text):
        return (await self.json("POST", "/tokenize", self.chat(text)))["tokens"]

    async def metrics(self):
        async with self.session.get(self.base + "/metrics", timeout=aiohttp.ClientTimeout(total=15)) as response:
            raw = await response.text()
            if response.status != 200:
                raise RuntimeError(f"/metrics: HTTP {response.status}")
        return parse_metrics(raw, self.args.model), raw

    async def reset(self):
        # This version returns 200 even if the engine did not reset. Preflight
        # verifies actual cold/hit/cold behavior; do not equate 200 with success.
        await self.json("POST", "/reset_prefix_cache")

    async def completion(self, text, request_id):
        payload = {**self.chat(text), "request_id": request_id, "stream": True,
                   "stream_options": {"include_usage": True}, "temperature": 0,
                   "max_tokens": self.args.output_tokens, "ignore_eos": True,
                   "skip_special_tokens": False, "logprobs": True, "top_logprobs": 0}
        start = time.perf_counter()
        first, usage, response_id, done = None, None, None, False
        async with self.session.post(self.base + "/v1/chat/completions", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"completion HTTP {response.status}: {(await response.text())[:1200]}")
            # aiohttp StreamReader iteration is line-oriented, not TCP-chunk-oriented.
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event or event.get("object") == "error":
                    raise RuntimeError(f"stream error: {event}")
                response_id = event.get("id", response_id)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    token_evidence = (choice.get("logprobs") or {}).get("content")
                    if first is None and (token_evidence or delta.get("content") or delta.get("reasoning_content")):
                        first = time.perf_counter()
        if not done or first is None or usage is None:
            raise RuntimeError("Incomplete SSE: missing [DONE], generated-token evidence, or final usage")
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        # Leave absent values unknown here. Only interpret absence as zero after
        # preflight proves this v0.9.2 server exposes positive cached_tokens.
        return {"response_id": response_id, "ttft_ms": (first - start) * 1000,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "cached_tokens_raw": cached, "usage": usage}


async def make_prefix(client, length, seed):
    rng = random.Random(seed)
    # ASCII avoids decode/re-encode ambiguity from truncated UTF-8 byte tokens.
    text = " ".join("".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=12))
                    for _ in range(length))
    ids = await client.tokens(prompt=text, add_special_tokens=False)
    if len(ids) < length:
        raise RuntimeError("Generated prefix is too short")
    ids = ids[:length]
    text = (await client.json("POST", "/detokenize", {"model": client.args.model, "tokens": ids}))["prompt"]
    check = await client.tokens(prompt=text, add_special_tokens=False)
    if check != ids:
        raise RuntimeError("Tokenizer round trip differs; cannot guarantee exact prefix length")
    return text


def suffix(seed, index):
    nonce = hashlib.sha256(f"suffix:{seed}:{index}".encode()).hexdigest()[:24]
    return "\n" + nonce + "\nReply with OK."


async def prepare(client, args, out):
    prefixes, references, reusable = [], [], []
    for i in range(args.prefixes):
        text = await make_prefix(client, args.prefix_tokens, f"pool:{args.pool_seed}:{i}")
        a = await client.chat_tokens(text + "\nAAA calibration A")
        b = await client.chat_tokens(text + "\nZZZ calibration B")
        end = lcp(a, b) // args.block_size * args.block_size
        prefixes.append(text)
        references.append(a[:end])
        reusable.append(end)
        if (i + 1) % 10 == 0 or i + 1 == args.prefixes:
            print(f"prepare prefixes {i + 1}/{args.prefixes}", flush=True)
    cross_shared = max((lcp(a, b) // args.block_size * args.block_size
                        for i, a in enumerate(references) for b in references[i + 1:]), default=0)
    if min(reusable) <= cross_shared or cross_shared >= args.prefix_tokens // 2:
        raise RuntimeError("Prefix pool has excessive inter-prefix sharing")
    weights = [(i + 1) ** (-args.alpha) for i in range(args.prefixes)]
    rng = random.Random(args.seed)
    sequence = rng.choices(range(args.prefixes), weights=weights, k=args.warmup + args.requests)
    trace = []
    for index, prefix_id in enumerate(sequence):
        tail = suffix(args.seed, index)
        ids = await client.chat_tokens(prefixes[prefix_id] + tail)
        end = reusable[prefix_id]
        if ids[:end] != references[prefix_id] or end > (len(ids) - 1) // args.block_size * args.block_size:
            raise RuntimeError(f"Unstable prefix boundary for request {index}")
        trace.append({"index": index, "phase": "warmup" if index < args.warmup else "measure",
                      "prefix_id": prefix_id, "suffix": tail, "expected_prompt_tokens": len(ids),
                      "reusable_prefix_tokens": end, "token_ids_sha256": digest(ids)})
        if (index + 1) % 200 == 0:
            print(f"validate prompts {index + 1}/{len(sequence)}", flush=True)
    save_json(out / "prefix_pool.json", {"pool_seed": args.pool_seed, "prefixes": prefixes,
                                        "reusable_prefix_tokens": reusable,
                                        "max_cross_prefix_shared_tokens": cross_shared})
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in trace:
            f.write(json.dumps(row) + "\n")
    workload = {"prefix_pool_sha256": digest(prefixes), "trace_sha256": digest(trace),
                "max_cross_prefix_shared_tokens": cross_shared,
                "nominal_prefix_pool_tokens": args.prefixes * args.prefix_tokens,
                "reusable_prefix_tokens_min": min(reusable), "reusable_prefix_tokens_max": max(reusable),
                "max_prompt_tokens": max(r["expected_prompt_tokens"] for r in trace),
                "probabilities": [w / sum(weights) for w in weights]}
    save_json(out / "workload.json", workload)
    return prefixes, trace, workload


async def settle(client, before, expected_queries, expected_success, timeout):
    start = time.perf_counter()
    while True:
        snapshot, raw = await client.metrics()
        ready = snapshot["queries"] is not None and snapshot["queries"] >= before["queries"] + expected_queries
        if before["success"] is not None:
            ready = ready and snapshot["success"] is not None and snapshot["success"] >= before["success"] + expected_success
        if ready and snapshot["running"] == 0 and snapshot["waiting"] == 0:
            return snapshot, raw
        if time.perf_counter() - start >= timeout:
            raise RuntimeError("Metrics did not catch up / server not idle. Check exclusive use, metrics and engine logs.")
        await asyncio.sleep(0.5)


async def preflight(client, args, out):
    before, _ = await client.metrics()
    for key in ("queries", "hits", "running", "waiting"):
        if before[key] is None:
            raise RuntimeError(f"Missing required Prometheus metric: {key}")
    if before["running"] or before["waiting"]:
        raise RuntimeError("Server is busy; use a dedicated idle server")
    prefix = await make_prefix(client, args.prefix_tokens, f"probe:{args.pool_seed}")
    prompt = prefix + "\nCache calibration. Reply with OK."
    ids = await client.chat_tokens(prompt)
    expected_hit = (len(ids) - 1) // args.block_size * args.block_size
    await client.reset()
    print("preflight: cold request", flush=True)
    cold = await client.completion(prompt, "calibration-cold")
    print("preflight: exact repeat", flush=True)
    warm = await client.completion(prompt, "calibration-hit")
    if warm["cached_tokens_raw"] is None or warm["cached_tokens_raw"] < expected_hit:
        save_json(out / "preflight.json", {"cold": cold, "warm": warm})
        raise RuntimeError("Cache calibration failed. Start V1 with --enable-prefix-caching "
                           "--enable-prompt-tokens-details; check block size / capacity.")
    await client.reset()
    print("preflight: cold after reset", flush=True)
    reset_cold = await client.completion(prompt, "calibration-reset")
    save_json(out / "preflight.json", {"cold": cold, "warm": warm, "reset_cold": reset_cold,
                                      "expected_repeat_hit_tokens": expected_hit})
    if (cold["cached_tokens_raw"] or 0) != 0 or (reset_cold["cached_tokens_raw"] or 0) != 0:
        raise RuntimeError("Reset did not produce zero cached tokens; aborting contaminated run")
    for row in (cold, warm, reset_cold):
        if row["prompt_tokens"] != len(ids) or row["completion_tokens"] != args.output_tokens:
            raise RuntimeError("Calibration usage differs from requested input/output lengths")
    snapshot, raw = await settle(client, before, sum(r["prompt_tokens"] for r in (cold, warm, reset_cold)),
                                 3, args.metrics_timeout)
    await client.reset()
    print(f"preflight passed: cold=0, repeat={warm['cached_tokens_raw']}, reset=0 cached tokens", flush=True)
    return snapshot, raw


async def monitor(client, out, interval, stop, samples):
    with (out / "metrics_timeseries.jsonl").open("w", encoding="utf-8") as f:
        while not stop.is_set():
            try:
                sample, _ = await client.metrics()
            except Exception as exc:
                sample = {"timestamp": time.time(), "error": str(exc)}
            samples.append(sample)
            f.write(json.dumps(sample) + "\n")
            f.flush()
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass


async def phase(client, args, out, prefixes, trace, rows, seen, completed, inflight):
    pending = iter(trace)
    stop = asyncio.Event()
    started = time.perf_counter()
    with (out / "requests.jsonl").open("a", encoding="utf-8") as f:
        async def worker():
            while not stop.is_set():
                entry = next(pending, None)
                if entry is None:
                    break
                p = entry["prefix_id"]
                row = {**entry, "request_id": f"zipf-{args.seed}-{entry['index']}",
                       "first_visit": seen[p] == 0, "prior_completed": completed[p] > 0,
                       "same_prefix_inflight_at_dispatch": inflight[p], "dispatch_time": time.time(),
                       "queue_ms": None, "status": "error"}
                seen[p] += 1
                inflight[p] += 1
                try:
                    result = await client.completion(prefixes[p] + entry["suffix"], row["request_id"])
                    row.update(result)
                    cached = result["cached_tokens_raw"] or 0
                    if result["prompt_tokens"] != entry["expected_prompt_tokens"]:
                        raise RuntimeError("Server prompt tokens differ from /tokenize")
                    if result["completion_tokens"] != args.output_tokens:
                        raise RuntimeError("Server output length differs from request")
                    if cached < 0 or cached > (result["prompt_tokens"] - 1) // args.block_size * args.block_size or cached % args.block_size:
                        raise RuntimeError("Invalid cached-token count; verify target version / block size")
                    retained = min(cached, entry["reusable_prefix_tokens"])
                    row.update(cached_tokens=cached, cached_prefix_tokens=retained,
                               full_prefix_hit=cached >= entry["reusable_prefix_tokens"],
                               prefix_recompute_tokens=entry["reusable_prefix_tokens"] - retained,
                               status="ok")
                    completed[p] += 1
                except Exception as exc:
                    row["error"] = str(exc)
                    stop.set()
                finally:
                    inflight[p] -= 1
                row["finish_time"] = time.time()
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if len(rows) % args.progress_every == 0 or row["status"] != "ok":
                    print(f"{entry['phase']} index={entry['index']} P{p} status={row['status']} "
                          f"cached={row.get('cached_tokens')} ttft_ms={row.get('ttft_ms', 0):.1f}", flush=True)
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    if stop.is_set():
        raise RuntimeError("Request failed; partial records preserved in requests.jsonl. No automatic retry.")
    return time.perf_counter() - started


def group_stats(rows):
    good = [r for r in rows if r["status"] == "ok"]
    mature = [r for r in good if r["prior_completed"]]
    full = [r for r in good if r["full_prefix_hit"]]
    partial = [r for r in good if not r["full_prefix_hit"]]
    saved = sum(r["cached_tokens"] for r in good)
    prompt = sum(r["prompt_tokens"] for r in good)
    return {"requests": len(rows), "successful_requests": len(good),
            "first_visit_requests": sum(r["first_visit"] for r in good),
            "prior_completed_requests": len(mature), "prompt_tokens": prompt,
            "saved_prefill_tokens": saved, "token_hit_rate": ratio(saved, prompt),
            "full_prefix_hit_rate": ratio(len(full), len(good)),
            "repeat_full_prefix_hit_rate": ratio(sum(r["full_prefix_hit"] for r in mature), len(mature)),
            "repeat_prefix_recompute_tokens": sum(r["prefix_recompute_tokens"] for r in mature),
            "same_prefix_overlap_requests": sum(r["same_prefix_inflight_at_dispatch"] > 0 for r in good),
            "ttft_ms_mean": statistics.mean(r["ttft_ms"] for r in good) if good else None,
            "ttft_ms_p50": percentile([r["ttft_ms"] for r in good], 0.5),
            "ttft_ms_p95": percentile([r["ttft_ms"] for r in good], 0.95),
            "full_hit_ttft_ms_p50": percentile([r["ttft_ms"] for r in full], 0.5),
            "non_full_hit_ttft_ms_p50": percentile([r["ttft_ms"] for r in partial], 0.5)}


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row if k not in ("usage", "suffix")))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


async def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "config.json", {**vars(args), "created_utc": datetime.now(timezone.utc).isoformat(),
                                    "python": sys.version,
                                    "note": "gpu_blocks is a user-declared capacity, not independently measured"})
    rows, samples, warnings = [], [], []
    headers = {}
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
    timeout = aiohttp.ClientTimeout(total=args.timeout, connect=30)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout,
                                    connector=aiohttp.TCPConnector(limit=max(args.concurrency + 4, 8)),
                                    read_bufsize=1024 * 1024) as session:
        client = Client(session, args)
        stop, monitor_task = asyncio.Event(), None
        try:
            version = await client.json("GET", "/version")
            if not str(version.get("version", "")).startswith("0.9.2"):
                raise RuntimeError(f"This script validates vLLM 0.9.2 semantics; server reports {version}")
            models = await client.json("GET", "/v1/models")
            cards = [m for m in models["data"] if m["id"] == args.model]
            if len(cards) != 1:
                raise RuntimeError(f"Model {args.model!r} not found; available: {[m['id'] for m in models['data']]}")
            try:
                info = await client.json("GET", "/server_info")
            except HTTPError as exc:
                if exc.status != 404:
                    raise
                info = {"available": False, "reason": "HTTP 404"}
                warnings.append("/server_info unavailable; preserve the server launch log to verify configuration.")
            save_json(out / "server.json", {"version": version, "models": models, "server_info": info})
            prefixes, trace, workload = await prepare(client, args, out)
            max_len = cards[0].get("max_model_len")
            if max_len and workload["max_prompt_tokens"] + args.output_tokens > max_len:
                raise RuntimeError("Input + output exceeds server max_model_len")
            baseline, raw = await preflight(client, args, out)
            save_json(out / "metrics_before_warmup.json", baseline)
            (out / "metrics_before_warmup.prom").write_text(raw, encoding="utf-8")
            monitor_task = asyncio.create_task(monitor(client, out, args.metrics_interval, stop, samples))
            seen, completed, inflight = ([0] * args.prefixes for _ in range(3))
            warm_trace = [r for r in trace if r["phase"] == "warmup"]
            await phase(client, args, out, prefixes, warm_trace, rows, seen, completed, inflight)
            warm_rows = list(rows)
            before, raw = await settle(client, baseline, sum(r["prompt_tokens"] for r in warm_rows),
                                       len(warm_rows), args.metrics_timeout)
            save_json(out / "metrics_before_measure.json", before)
            (out / "metrics_before_measure.prom").write_text(raw, encoding="utf-8")
            print("measurement starts (warmup excluded)", flush=True)
            seconds = await phase(client, args, out, prefixes, [r for r in trace if r["phase"] == "measure"],
                                  rows, seen, completed, inflight)
            measured = [r for r in rows if r["phase"] == "measure"]
            after, raw = await settle(client, before, sum(r["prompt_tokens"] for r in measured),
                                      len(measured), args.metrics_timeout)
            save_json(out / "metrics_after_measure.json", after)
            (out / "metrics_after_measure.prom").write_text(raw, encoding="utf-8")
            delta = {k: after[k] - before[k] if after[k] is not None and before[k] is not None else None
                     for k in ("queries", "hits", "preemptions", "success", "queue_seconds_sum", "queue_seconds_count")}
            stats = group_stats(measured)
            if delta["queries"] != stats["prompt_tokens"] or delta["hits"] != stats["saved_prefill_tokens"]:
                warnings.append("Prometheus and per-request totals differ: inspect retries, preemption, external traffic or patched server.")
            if delta["preemptions"] is None:
                warnings.append("Preemption counter unavailable; absence of preemption is unverified.")
            elif delta["preemptions"] > 0:
                warnings.append("Preemption occurred; cached tokens do not account for all subsequent recomputation.")
            if args.concurrency > 1:
                warnings.append("Closed-loop concurrency: submission order is recorded; execution order and arrival times differ across runs.")
            per_prefix = [{"prefix_id": i, **group_stats([r for r in measured if r["prefix_id"] == i])}
                          for i in range(args.prefixes)]
            summary = {"status": "complete", "config": vars(args), "workload": workload,
                       "measurement": {**stats, "wall_seconds": seconds,
                                       "requests_per_second": len(measured) / seconds,
                                       "output_tokens_per_second": sum(r["completion_tokens"] for r in measured) / seconds},
                       "top10": group_stats([r for r in measured if r["prefix_id"] < 10]),
                       "prometheus_delta": delta, "warnings": warnings,
                       "interpretation": "repeat means a previous request of the same prefix completed before dispatch; misses alone do not prove policy suboptimality"}
            save_json(out / "summary.json", summary)
            write_csv(out / "per_prefix.csv", per_prefix)
            print(json.dumps({"output": str(out.resolve()), **stats, "warnings": warnings}, indent=2), flush=True)
        except Exception as exc:
            save_json(out / "failure.json", {"error": str(exc), "completed_rows": len(rows)})
            raise
        finally:
            stop.set()
            if monitor_task:
                await monitor_task
            write_csv(out / "requests.csv", sorted(rows, key=lambda r: r["index"]))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--out", required=True, help="New output directory; existing directories are never overwritten")
    parser.add_argument("--prefixes", type=int, default=100)
    parser.add_argument("--prefix-tokens", type=int, default=3072, help="Exact raw prefix body length; template adds tokens")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Zipf trace and unique suffix seed")
    parser.add_argument("--pool-seed", type=int, default=20261013, help="Independent fixed prefix-pool seed")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--gpu-blocks", type=int, default=None, help="Record capacity from server launch/log; does not configure server")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--metrics-timeout", type=float, default=60)
    parser.add_argument("--metrics-interval", type=float, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    for name in ("prefixes", "prefix_tokens", "block_size", "requests", "concurrency", "output_tokens", "timeout",
                 "metrics_timeout", "metrics_interval", "progress_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0 or not math.isfinite(args.alpha) or args.alpha < 0:
        parser.error("warmup and finite alpha must be nonnegative")
    if args.prefix_tokens < args.block_size or args.prefix_tokens % args.block_size:
        parser.error("prefix-tokens must be a positive multiple of block-size")
    if args.gpu_blocks is not None and args.gpu_blocks <= 0:
        parser.error("gpu-blocks must be positive")
    return args


if __name__ == "__main__":
    try:
        asyncio.run(run(arguments()))
    except KeyboardInterrupt:
        print("Interrupted; already written JSONL records are preserved.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
