import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("comparison", ROOT / "compare_results.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def run_fixture(self):
        return {"status": "complete", "workload": {"prefix_pool_sha256": "pool", "trace_sha256": "trace"},
                "config": {"model": "qwen3-4b", "prefixes": 100, "prefix_tokens": 3072,
                           "block_size": 16, "alpha": 1.0, "seed": 42, "pool_seed": 1,
                           "warmup": 20, "requests": 100, "concurrency": 1, "output_tokens": 1,
                           "gpu_blocks": 2048},
                "measurement": {"successful_requests": 100, "repeat_prefix_recompute_tokens": 100,
                                "repeat_full_prefix_hit_rate": 0.5, "ttft_ms_mean": 100}}

    def test_reduction_and_percentage_points(self):
        a = self.run_fixture()
        b = copy.deepcopy(a)
        b["measurement"]["repeat_prefix_recompute_tokens"] = 50
        b["measurement"]["repeat_full_prefix_hit_rate"] = 0.6
        rows = {r[0]: r for r in comparison.compare(a, b)}
        self.assertEqual(rows["repeat_prefix_recompute_tokens"][3], "-50.00%")
        self.assertEqual(rows["repeat_full_prefix_hit_rate"][3], "+10.00 pp")

    def test_refuse_mismatched_trace_capacity_and_failures(self):
        a = self.run_fixture()
        for category, field, value in (("workload", "trace_sha256", "other"),
                                       ("config", "gpu_blocks", 4096),
                                       ("measurement", "successful_requests", 99)):
            b = copy.deepcopy(a)
            b[category][field] = value
            with self.assertRaises(ValueError):
                comparison.compare(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
