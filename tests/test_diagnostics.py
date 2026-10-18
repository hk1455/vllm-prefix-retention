import argparse
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
run = load('aba', ROOT / 'diagnostics/run_aba.py')
bench = load('bench', ROOT / 'zipf_prefix_bench.py')

class Guards(unittest.TestCase):
    def test_actual_policy_and_capacity(self):
        text = '\x1b[32mPREFIX_RETENTION policy=slru protected_ratio=0.5\nGPU KV cache size: 32,768 tokens'
        self.assertEqual(run.verify_log(text, 'slru', 2048)['kv_tokens'], 32768)

    def test_wrong_policy_or_capacity_refused(self):
        for text in ('PREFIX_RETENTION policy=lru\nGPU KV cache size: 32,768 tokens',
                     'PREFIX_RETENTION policy=slru\nGPU KV cache size: 65,536 tokens',
                     'policy=slru blocks=2048'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                run.verify_log(text, 'slru', 2048)

    def test_cold_hit_and_length_refused(self):
        for row in ({'cached_tokens':16,'prompt_tokens':3072,'completion_tokens':1},
                    {'cached_tokens':0,'prompt_tokens':3071,'completion_tokens':1},
                    {'cached_tokens':0,'prompt_tokens':3072,'completion_tokens':2}):
            with self.subTest(row=row), self.assertRaises(RuntimeError):
                run.validate_cold(row, 3072)

    def test_active_listener_not_reused(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            sock.listen()
            with self.assertRaises(RuntimeError):
                run.require_free_port(sock.getsockname()[1])


class HTTPProbe(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp import web
        self.web = web
        self.last = None
        self.ignore_reset = False
        self.no_hit_reporting = False
        self.queries = self.hits = self.success = 0
        app = web.Application()
        app.add_routes([web.get('/version', self.version), web.post('/tokenize', self.tokenize),
                        web.post('/detokenize', self.detokenize), web.post('/reset_prefix_cache', self.reset),
                        web.get('/metrics', self.metrics), web.post('/v1/chat/completions', self.completion)])
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.args = argparse.Namespace(base_url=f'http://127.0.0.1:{port}', cold_warmup=1, cold_requests=3)
        self.temp = tempfile.TemporaryDirectory()

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.temp.cleanup()

    def ids(self, body):
        if 'messages' in body:
            return [1,2] + [ord(c) for m in body['messages'] for c in m['content']]
        return [ord(c) for c in body['prompt']]

    async def version(self, request):
        return self.web.json_response({'version':'0.9.2'})

    async def tokenize(self, request):
        return self.web.json_response({'tokens':self.ids(await request.json())})

    async def detokenize(self, request):
        return self.web.json_response({'prompt':''.join(chr(c) for c in (await request.json())['tokens'])})

    async def reset(self, request):
        if not self.ignore_reset:
            self.last = None
        return self.web.Response(status=200)

    async def metrics(self, request):
        metrics = {'prefix_cache_queries_total':self.queries, 'prefix_cache_hits_total':self.hits,
                   'request_success_total':self.success, 'num_requests_running':0, 'num_requests_waiting':0}
        return self.web.Response(text='\n'.join(f'vllm:{k} {v}' for k,v in metrics.items()))

    async def completion(self, request):
        ids = self.ids(await request.json())
        cached = ((len(ids)-1)//16)*16 if self.last == ids else 0
        self.last = ids
        self.queries += len(ids)
        self.hits += cached
        self.success += 1
        usage = {'prompt_tokens':len(ids), 'completion_tokens':1,
                 'prompt_tokens_details':{'cached_tokens':0 if self.no_hit_reporting else cached}}
        events = [{'choices':[{'delta':{'role':'assistant'}}]},
                  {'choices':[{'delta':{'content':'OK'}}]}, {'choices':[], 'usage':usage}]
        text = ''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n'
        return self.web.Response(text=text, content_type='text/event-stream')

    async def test_stream_probe_saves_measured_only_summary(self):
        out = Path(self.temp.name)/'cold'
        result = await run.cold_probe(bench,self.args,out)
        self.assertEqual(result['requests'],3)
        self.assertTrue(result['all_cached_tokens_zero'])
        rows = [json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),4)
        self.assertEqual(sum(r['phase']=='measure' for r in rows),3)
        self.assertTrue(all(r['reset_end']<=r['dispatch_time']<=r['finish_time'] for r in rows))
        self.assertEqual(len(set(r['token_ids_sha256'] for r in rows)),1)

    async def test_ignored_reset_refused(self):
        self.ignore_reset = True
        with self.assertRaisesRegex(RuntimeError,'Reset did not produce zero'):
            await run.cold_probe(bench,self.args,Path(self.temp.name)/'cold')

    async def test_missing_positive_hit_refused(self):
        self.no_hit_reporting = True
        with self.assertRaisesRegex(RuntimeError,'Cache calibration failed'):
            await run.cold_probe(bench,self.args,Path(self.temp.name)/'cold')

if __name__=='__main__':
    unittest.main()
