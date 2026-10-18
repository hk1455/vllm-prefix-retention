#!/usr/bin/env python3
"""Cache-behavior check with separate output diagnostics. Clears prefix cache."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import urllib.request


def check_server(call, model, max_tokens=1, block_size=16):
    body = {
        "model": model,
        "messages": [{"role": "user", "content":
            ("Reference rule: the service checks its prefix cache before computing a response.\n" * 100)
            + "Reply with exactly: cache test passed"}],
        "temperature": 0, "seed": 42, "max_tokens": max_tokens, "ignore_eos": False,
        "logprobs": True, "top_logprobs": 2,
        "stream": False, "chat_template_kwargs": {"enable_thinking": False},
    }
    report = {"probe_version": 2, "request": body, "block_size": block_size,
              "responses": [], "cache_behavior_pass": False, "output_match": None,
              "first_token_match": None, "cleanup_reset_requested": False,
              "errors": [], "warnings": []}
    try:
        call("/reset_prefix_cache", None)
        for phase in ("cold", "hit", "reset_cold"):
            if phase == "reset_cold":
                call("/reset_prefix_cache", None)
            response = call("/v1/chat/completions", body)
            usage = response["usage"]
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            prompt = usage["prompt_tokens"]
            if type(prompt) is not int or prompt <= block_size:
                raise ValueError(f"Invalid/too short prompt_tokens: {prompt!r}")
            if type(cached) is not int or not 0 <= cached <= prompt:
                raise ValueError(f"Invalid cached_tokens: {cached!r}")
            choice = response["choices"][0]
            message = choice["message"]
            tokens = (choice.get("logprobs") or {}).get("content") or []
            record = {
                "phase": phase, "cached_tokens": cached, "prompt_tokens": prompt,
                "completion_tokens": usage.get("completion_tokens"),
                "output": [message.get("content"), message.get("reasoning_content")],
                "finish_reason": choice.get("finish_reason"),
                "first_token": ({"token": tokens[0].get("token"),
                                 "bytes": tokens[0].get("bytes")} if tokens else None),
                "response": response,
            }
            report["responses"].append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "response"}, ensure_ascii=False))
        cold, hit, reset = report["responses"]
        # Full attention / no speculation: keep the last prompt token for logits.
        expected = ((cold["prompt_tokens"] - 1) // block_size) * block_size
        report["expected_hit_tokens"] = expected
        if len({r["prompt_tokens"] for r in report["responses"]}) != 1:
            report["errors"].append("Prompt token counts differ")
        if cold["cached_tokens"] != 0 or reset["cached_tokens"] != 0:
            report["errors"].append("Cold/reset request did not report zero cached tokens")
        if hit["cached_tokens"] != expected:
            report["errors"].append(f"Expected {expected} cached tokens on repeat, got {hit['cached_tokens']}")
        report["cache_behavior_pass"] = not report["errors"]
        report["output_match"] = cold["output"] == hit["output"] == reset["output"]
        report["cold_output_match"] = cold["output"] == reset["output"]
        if all(r["first_token"] is not None for r in report["responses"]):
            report["first_token_match"] = cold["first_token"] == hit["first_token"] == reset["first_token"]
        else:
            report["warnings"].append("First-token logprobs unavailable; token identity not verified")
        if not report["output_match"] or report["first_token_match"] is False:
            report["warnings"].append("Cold/hit output differs. Cache counters do not prove numerical correctness; keep this report and compare LRU/SLRU.")
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            call("/reset_prefix_cache", None)
            # This last HTTP ACK alone does not prove reset; reset-cold above does.
            report["cleanup_reset_requested"] = True
        except Exception as exc:
            report["errors"].append(f"Cleanup reset failed: {type(exc).__name__}: {exc}")
    report["status"] = ("error" if report["errors"] else
                        "cache_pass_output_warning" if report["warnings"] else "cache_pass_output_match")
    return report


def exit_code(report, strict_output=False):
    if report["errors"] or not report["cache_behavior_pass"]:
        return 1
    if strict_output and (report["output_match"] is not True or report["first_token_match"] is not True):
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--strict-output", action="store_true",
                        help="Exit 2 on output/token mismatch; cache/protocol errors always exit 1")
    parser.add_argument("--label", choices=("lru", "slru", "unknown"), default="unknown",
                        help="User label only; verify the engine log")
    parser.add_argument("--out", type=Path, help="New JSON report path; never overwrite")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.block_size < 1:
        parser.error("max-tokens and block-size must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = args.out or Path(f"smoke_{args.label}_{stamp}.json")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]

    def call(route, body):
        data = json.dumps(body).encode() if body is not None else b""
        req = urllib.request.Request(base + route, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=300) as response:
            raw = response.read()
        return json.loads(raw) if raw.strip() else None

    report = check_server(call, args.model, args.max_tokens, args.block_size)
    report.update(label=args.label, label_verified_by_server=False, created_utc=stamp,
                  scope="Cache API behavior and output diagnostics; not a complete KV correctness test")
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({k: report[k] for k in (
        "status", "cache_behavior_pass", "output_match", "first_token_match", "errors", "warnings")},
        ensure_ascii=False, indent=2))
    print(f"Report: {output.resolve()}")
    return exit_code(report, args.strict_output)


if __name__ == "__main__":
    raise SystemExit(main())
