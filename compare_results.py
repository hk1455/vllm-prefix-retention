#!/usr/bin/env python3
"""Compare paired runs, refusing different traces or client configurations."""
import argparse
import json
from pathlib import Path


def read_run(path):
    path = Path(path)
    return json.loads((path / "summary.json" if path.is_dir() else path).read_text(encoding="utf-8"))


def compare(a, b):
    for x in (a, b):
        if x.get("status") != "complete":
            raise ValueError("Both summary.json files must have status=complete")
    for k in ("prefix_pool_sha256", "trace_sha256"):
        if not a["workload"].get(k) or a["workload"].get(k) != b["workload"].get(k):
            raise ValueError(f"Workloads differ or fingerprint missing: {k}")
    keys = ("model", "prefixes", "prefix_tokens", "block_size", "alpha", "seed",
            "pool_seed", "warmup", "requests", "concurrency", "output_tokens", "gpu_blocks")
    for k in keys:
        if k not in a["config"] or k not in b["config"] or a["config"][k] != b["config"][k]:
            raise ValueError(f"Client configurations differ or field missing: {k}")
    if a["config"]["gpu_blocks"] is None:
        raise ValueError("Capacity label missing; rerun with --gpu-blocks")
    am, bm = a["measurement"], b["measurement"]
    for name, run, m in (("baseline", a, am), ("candidate", b, bm)):
        if m["successful_requests"] != run["config"]["requests"]:
            raise ValueError(f"Failed/incomplete requests in {name}")
    metrics = ["repeat_prefix_recompute_tokens", "saved_prefill_tokens", "repeat_full_prefix_hit_rate",
               "token_hit_rate", "ttft_ms_mean", "ttft_ms_p50", "ttft_ms_p95", "requests_per_second"]
    result = []
    for k in metrics:
        av, bv = am.get(k), bm.get(k)
        if av is None or bv is None:
            change = "n/a"
        elif k.endswith("rate"):
            change = f"{100 * (bv-av):+.2f} pp"
        else:
            change = f"{100 * (bv-av) / av:+.2f}%" if av else "baseline=0"
        result.append((k, av, bv, change))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline")
    p.add_argument("candidate")
    args = p.parse_args()
    a, b = read_run(args.baseline), read_run(args.candidate)
    rows = compare(a, b)
    print("Client trace/configuration matched. Verify actual policy/capacity in SERVER logs.")
    print("Changes are candidate minus baseline (negative latency/recompute is better).")
    print(f"{'metric':36} {'baseline':>16} {'candidate':>16} {'change':>14}")
    for k, av, bv, delta in rows:
        fmt = lambda v: "n/a" if v is None else f"{v:.6g}"
        print(f"{k:36} {fmt(av):>16} {fmt(bv):>16} {delta:>14}")
    for name, x in (("baseline", a), ("candidate", b)):
        for warning in x.get("warnings", []):
            print(f"WARNING [{name}]: {warning}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"Comparison refused: {exc}")
