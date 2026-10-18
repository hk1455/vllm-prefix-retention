import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("smoke", ROOT / "smoke_server.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class FakeAPI:
    def __init__(self, mismatch=False, token_mismatch=False, reset_broken=False,
                 missing_details=False, fail=False, partial=False, cleanup_fail=False):
        self.mismatch, self.token_mismatch = mismatch, token_mismatch
        self.reset_broken, self.missing_details = reset_broken, missing_details
        self.fail, self.partial, self.cleanup_fail = fail, partial, cleanup_fail
        self.calls, self.resets = [], 0
        self.warm = False

    def __call__(self, route, body):
        self.calls.append((route, body))
        if route == "/reset_prefix_cache":
            self.resets += 1
            if self.cleanup_fail and self.resets == 3:
                raise RuntimeError("cleanup error")
            if not self.reset_broken:
                self.warm = False
            return None
        if self.fail:
            raise RuntimeError("HTTP 500")
        cached = (1392 if self.partial else 1408) if self.warm else 0
        text = "cache differs" if self.mismatch and self.warm else "cache test passed"
        token = "other" if self.token_mismatch and self.warm else "cache"
        self.warm = True
        usage = {"prompt_tokens": 1415, "completion_tokens": 1}
        if cached and not self.missing_details:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        return {"choices": [{"message": {"content": text}, "finish_reason": "length",
                             "logprobs": {"content": [{"token": token, "bytes": list(token.encode()),
                                                       "logprob": -0.01}]}}], "usage": usage}


class SmokeTests(unittest.TestCase):
    def run_check(self, fake):
        with contextlib.redirect_stdout(io.StringIO()):
            return smoke.check_server(fake, "qwen3-4b")

    def test_cache_contract_single_token_eos_default(self):
        fake = FakeAPI()
        r = self.run_check(fake)
        self.assertTrue(r["cache_behavior_pass"])
        self.assertTrue(r["output_match"])
        self.assertTrue(r["first_token_match"])
        self.assertEqual(r["expected_hit_tokens"], 1408)
        self.assertEqual(smoke.exit_code(r, True), 0)
        requests = [b for route, b in fake.calls if route == "/v1/chat/completions"]
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(b["max_tokens"] == 1 and not b["ignore_eos"] for b in requests))
        self.assertEqual(fake.resets, 3)

    def test_text_mismatch_preserved_and_strict_exit(self):
        r = self.run_check(FakeAPI(mismatch=True))
        self.assertTrue(r["cache_behavior_pass"])
        self.assertFalse(r["output_match"])
        self.assertEqual(r["status"], "cache_pass_output_warning")
        self.assertEqual(smoke.exit_code(r), 0)
        self.assertEqual(smoke.exit_code(r, True), 2)
        self.assertEqual(len(r["responses"]), 3)
        self.assertIn("response", r["responses"][1])

    def test_different_first_token_same_text_is_warning(self):
        r = self.run_check(FakeAPI(token_mismatch=True))
        self.assertTrue(r["output_match"])
        self.assertFalse(r["first_token_match"])
        self.assertEqual(r["status"], "cache_pass_output_warning")
        self.assertEqual(smoke.exit_code(r, True), 2)

    def test_broken_reset_fails_even_with_http_200(self):
        r = self.run_check(FakeAPI(reset_broken=True))
        self.assertFalse(r["cache_behavior_pass"])
        self.assertEqual(smoke.exit_code(r), 1)

    def test_missing_cached_details_cannot_pass(self):
        r = self.run_check(FakeAPI(missing_details=True))
        self.assertFalse(r["cache_behavior_pass"])
        self.assertEqual(smoke.exit_code(r), 1)

    def test_partial_hit_fails_controlled_repeat(self):
        r = self.run_check(FakeAPI(partial=True))
        self.assertFalse(r["cache_behavior_pass"])
        self.assertEqual(smoke.exit_code(r), 1)

    def test_failed_request_still_attempts_cleanup(self):
        fake = FakeAPI(fail=True)
        r = self.run_check(fake)
        self.assertEqual(smoke.exit_code(r), 1)
        self.assertTrue(r["cleanup_reset_requested"])
        self.assertEqual(fake.resets, 2)

    def test_cleanup_error_not_silently_ignored(self):
        r = self.run_check(FakeAPI(cleanup_fail=True))
        self.assertEqual(smoke.exit_code(r), 1)
        self.assertFalse(r["cleanup_reset_requested"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
