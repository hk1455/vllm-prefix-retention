import asyncio
from collections import OrderedDict
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench", ROOT / "zipf_prefix_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class FakeVLLM:
    def __init__(self, capacity=32, expose=True, broken_reset=False, fail_index=None, info_status=200):
        self.capacity, self.expose, self.broken_reset, self.fail_index = capacity, expose, broken_reset, fail_index
        self.info_status = info_status
        self.cache = OrderedDict()
        self.queries = self.hits = self.success = self.running = 0
        self.pending = []

    def tokenize(self, body):
        if "messages" in body:
            return [200, 201, 202, 203] + list(body["messages"][0]["content"].encode()) + [204, 205]
        return list(body["prompt"].encode())

    async def handle(self, request):
        route = request.path
        if route == "/version":
            return web.json_response({"version": "0.9.2"})
        if route == "/server_info":
            if self.info_status != 200:
                return web.json_response({"detail": "test status"}, status=self.info_status)
            return web.json_response({"vllm_config": "fake-test-server"})
        if route == "/v1/models":
            return web.json_response({"data": [{"id": "qwen3-4b", "max_model_len": 4096}]})
        if route == "/metrics":
            lines = []
            for k, v in {"prefix_cache_queries_total": self.queries, "prefix_cache_hits_total": self.hits,
                         "num_preemptions_total": 0, "num_requests_running": self.running,
                         "num_requests_waiting": 0, "kv_cache_usage_perc": 0,
                         "request_success_total": self.success,
                         "gpu_prefix_cache_hits_total": 999999,
                         "gpu_prefix_cache_queries_total": 999999}.items():
                lines.append(f'vllm:{k}{{model_name="qwen3-4b"}} {v}')
            lines.append('vllm:prefix_cache_hits_total{model_name="different-model"} 123456')
            return web.Response(text="\n".join(lines))
        body = await request.json() if request.can_read_body else {}
        if route == "/reset_prefix_cache":
            if not self.broken_reset:
                self.cache.clear()
            return web.Response(status=200)
        if route == "/tokenize":
            tokens = self.tokenize(body)
            return web.json_response({"count": len(tokens), "tokens": tokens, "max_model_len": 4096})
        if route == "/detokenize":
            return web.json_response({"prompt": bytes(body["tokens"]).decode()})
        if route == "/v1/chat/completions":
            if self.fail_index is not None and body["request_id"] == f"zipf-42-{self.fail_index}":
                return web.json_response({"error": "intentional mock failure"}, status=500)
            tokens = self.tokenize(body)
            cached = 0
            for end in range(16, len(tokens), 16):
                key = tuple(tokens[:end])
                if key not in self.cache:
                    break
                self.cache.move_to_end(key)
                cached = end
            self.running += 1
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)

            async def emit(event):
                data = ("data: " + json.dumps(event) + "\n\n").encode()
                # Exercise fragmented packets, including a split JSON line.
                await response.write(data[:11])
                await asyncio.sleep(0)
                await response.write(data[11:])

            await emit({"id": body["request_id"], "choices": [{"delta": {"role": "assistant", "content": ""}}]})
            await asyncio.sleep(0.025)
            await emit({"id": body["request_id"], "choices": [{"delta": {"content": ""},
                        "logprobs": {"content": [{"token": "", "logprob": 0}]}, "finish_reason": "length"}]})
            usage = {"prompt_tokens": len(tokens), "completion_tokens": body["max_tokens"]}
            if cached and self.expose:
                usage["prompt_tokens_details"] = {"cached_tokens": cached}
            await emit({"choices": [], "usage": usage})
            # Approximate a capacity-limited prefix block store; not a vLLM simulator.
            for end in range(16, len(tokens) + 1, 16):
                self.cache[tuple(tokens[:end])] = True
                self.cache.move_to_end(tuple(tokens[:end]))
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
            self.running -= 1
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()

            async def publish():
                await asyncio.sleep(0.08)
                self.queries += len(tokens)
                self.hits += cached
                self.success += 1
            self.pending.append(asyncio.create_task(publish()))
            return response
        return web.Response(status=404)


class BenchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runners = []

    async def asyncTearDown(self):
        for runner, fake in self.runners:
            await asyncio.gather(*fake.pending)
            await runner.cleanup()
        self.temp.cleanup()

    async def execute(self, name, fake=None, concurrency=1):
        fake = fake or FakeVLLM()
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", fake.handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.runners.append((runner, fake))
        out = Path(self.temp.name) / name
        args = SimpleNamespace(base_url=f"http://127.0.0.1:{port}", model="qwen3-4b", out=str(out),
                               prefixes=4, prefix_tokens=64, block_size=16, alpha=1., seed=42,
                               pool_seed=123, warmup=3, requests=20, concurrency=concurrency,
                               output_tokens=1, gpu_blocks=fake.capacity, timeout=5,
                               metrics_timeout=3, metrics_interval=.04, progress_every=100)
        with redirect_stdout(io.StringIO()):
            await bench.run(args)
        return out, args

    async def test_serial_metrics_and_retention(self):
        out, args = await self.execute("serial")
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["measurement"]["requests"], 20)
        self.assertEqual(summary["prometheus_delta"]["queries"], summary["measurement"]["prompt_tokens"])
        self.assertEqual(summary["prometheus_delta"]["hits"], summary["measurement"]["saved_prefill_tokens"])
        self.assertGreater(summary["measurement"]["saved_prefill_tokens"], 0)
        self.assertEqual(summary["warnings"], [])
        records = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 23)
        self.assertTrue(all(r["ttft_ms"] >= 20 for r in records))
        self.assertEqual([r["index"] for r in records], list(range(23)))
        self.assertTrue(all(r["cached_tokens"] == 0 for r in records if r["first_visit"]))
        with self.assertRaises(FileExistsError):
            await bench.run(args)

    async def test_server_info_404_is_optional(self):
        out, _ = await self.execute("no-server-info", FakeVLLM(info_status=404))
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(any("/server_info unavailable" in w for w in summary["warnings"]))

    async def test_server_info_other_errors_are_fatal(self):
        with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
            await self.execute("broken-server-info", FakeVLLM(info_status=500))

    async def test_trace_reproducible_across_capacity_and_concurrency(self):
        a, _ = await self.execute("small", FakeVLLM(capacity=12))
        b, _ = await self.execute("large", FakeVLLM(capacity=128), concurrency=3)
        wa = json.loads((a / "workload.json").read_text())
        wb = json.loads((b / "workload.json").read_text())
        self.assertEqual(wa["trace_sha256"], wb["trace_sha256"])
        self.assertEqual(wa["prefix_pool_sha256"], wb["prefix_pool_sha256"])
        sa = json.loads((a / "summary.json").read_text())
        sb = json.loads((b / "summary.json").read_text())
        self.assertGreater(sa["measurement"]["repeat_prefix_recompute_tokens"], 0)
        self.assertGreater(sb["measurement"]["same_prefix_overlap_requests"], 0)
        self.assertEqual(sb["measurement"]["requests"], 20)
        self.assertEqual(sb["prometheus_delta"]["hits"], sb["measurement"]["saved_prefill_tokens"])

    async def test_missing_details_is_failure_not_zero_hit_result(self):
        with self.assertRaisesRegex(RuntimeError, "Cache calibration failed"):
            await self.execute("no-details", FakeVLLM(expose=False))
        self.assertTrue((Path(self.temp.name) / "no-details" / "failure.json").exists())
        self.assertFalse((Path(self.temp.name) / "no-details" / "summary.json").exists())

    async def test_reset_200_without_engine_reset_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Reset did not produce zero"):
            await self.execute("bad-reset", FakeVLLM(broken_reset=True))

    async def test_partial_request_failure_keeps_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "Request failed"):
            await self.execute("http-error", FakeVLLM(fail_index=5))
        out = Path(self.temp.name) / "http-error"
        rows = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(rows[-1]["status"], "error")
        self.assertIn("HTTP 500", rows[-1]["error"])
        self.assertTrue((out / "requests.csv").exists())
        self.assertFalse((out / "summary.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
