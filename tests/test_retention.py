#!/usr/bin/env python3
"""CPU tests of uploaded BlockPool and actual block/queue classes.

Default: extract block classes from the uploaded source; stub ONLY unrelated
vLLM dependencies. --installed: import the installed vLLM modules instead.
Neither mode executes attention kernels or measures Jetson performance.
"""
import ast
from collections import OrderedDict
from contextlib import contextmanager
import dataclasses
import importlib.util
import logging
import os
from pathlib import Path
import random
import sys
import types
import typing
import unittest

ROOT = Path(__file__).resolve().parents[1]
INSTALLED = "--installed" in sys.argv
if INSTALLED:
    sys.argv.remove("--installed")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extracted_function(path, cls, method, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method]


if INSTALLED:
    from vllm.v1.core import block_pool as patched
    from vllm.v1.core import kv_cache_utils as utils
    from vllm.v1.core import prefix_retention as policy_module
    original = load("retention_original_pool", ROOT / "reference/vllm/v1/core/block_pool.py")
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    find_hit = FullAttentionManager.find_longest_cache_hit
else:
    # Use the real uploaded dataclass and queue, not a reimplementation.
    utils = types.ModuleType("vllm.v1.core.kv_cache_utils")
    sys.modules[utils.__name__] = utils
    utils.__dict__.update(vars(typing))
    utils.__dict__["__name__"] = "vllm.v1.core.kv_cache_utils"
    utils.dataclass = dataclasses.dataclass
    source = ROOT / "reference/vllm/v1/core/kv_cache_utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and
             n.name in ("BlockHash", "KVCacheBlock", "FreeKVCacheBlockQueue")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), vars(utils))
    # Tests supply precomputed hashes. Unexpected use of hash computation fails.
    def unexpected(*args, **kwargs):
        raise AssertionError("Unexpected hash computation in snapshot harness")
    utils.generate_block_hash_extra_keys = utils.hash_block_tokens = unexpected
    events = types.ModuleType("vllm.distributed.kv_events")
    class Event:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    for n in ("AllBlocksCleared", "BlockRemoved", "BlockStored", "KVCacheEvent"):
        setattr(events, n, type(n, (Event,), {}))
    sys.modules[events.__name__] = events
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    sys.modules[logger.__name__] = logger
    request = types.ModuleType("vllm.v1.request")
    request.Request = object
    sys.modules[request.__name__] = request
    policy_module = load("vllm.v1.core.prefix_retention", ROOT / "payload/prefix_retention.py")
    original = load("retention_original_pool", ROOT / "reference/vllm/v1/core/block_pool.py")
    patched = load("retention_patched_pool", ROOT / "payload/block_pool.py")
    find_hit = extracted_function(
        ROOT / "reference/vllm/v1/core/single_type_kv_cache_manager.py",
        "FullAttentionManager", "find_longest_cache_hit", vars(utils).copy())


@contextmanager
def config(policy="slru", ratio="0.5"):
    updates = {"VLLM_PREFIX_RETENTION_POLICY": policy,
               "VLLM_PREFIX_RETENTION_PROTECTED_RATIO": ratio}
    previous = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def pool(n=9, policy="slru", ratio="0.5", caching=True, events=False, cls=None):
    with config(policy, ratio):
        return (cls or patched.BlockPool)(n, caching, events)


def hashes(key, n):
    # Distinct deterministic identities stand in for precomputed chained hashes.
    return [utils.BlockHash(key * 10000 + i, (key, i), None) for i in range(n)]


def cache_blocks(p, blocks, key):
    hs = hashes(key, len(blocks))
    req = types.SimpleNamespace(all_token_ids=[key] * (len(blocks) * 16),
                                request_id=str(key), lora_request=None)
    p.cache_full_blocks(req, blocks, hs, 0, len(blocks), 16, hash)
    return hs


def release(p, blocks):
    # Exact order supplied by the real SingleTypeKVCacheManager.free().
    p.free_blocks(reversed(blocks))


def match(p, hs):
    manager = types.SimpleNamespace(block_pool=p, block_size=16, use_eagle=False)
    return find_hit(manager, hs, len(hs) * 16)


class RuntimeTests(unittest.TestCase):
    def invariant(self, p):
        queue = p.free_block_queue.get_all_free_blocks()
        ids = [b.block_id for b in queue]
        expected = {b.block_id for b in p.blocks if b.ref_cnt == 0 and not b.is_null}
        self.assertEqual(set(ids), expected)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), p.get_num_free_blocks())
        for b in p.blocks:
            self.assertGreaterEqual(b.ref_cnt, 0)
            if b.block_hash is not None:
                self.assertIs(p.cached_block_hash_to_block[b.block_hash][b.block_id], b)
        for h, by_id in p.cached_block_hash_to_block.items():
            self.assertTrue(by_id)
            for bid, b in by_id.items():
                self.assertEqual(h, b.block_hash)
                self.assertEqual(bid, b.block_id)
        q = getattr(p, "_retention_queue", None)
        if q is not None:
            self.assertLessEqual(len(q.protected), q.protected_limit)

    def test_initialization_exhaustion_and_null(self):
        for mode in ("lru", "slru"):
            p = pool(6, mode)
            self.assertEqual(p.null_block.block_id, 0)
            blocks = p.get_new_blocks(5)
            self.assertEqual([b.block_id for b in blocks], [1, 2, 3, 4, 5])
            with self.assertRaises(ValueError):
                p.get_new_blocks(1)
            release(p, blocks)
            self.invariant(p)
            self.assertNotIn(p.null_block, p.free_block_queue.get_all_free_blocks())

    def test_lookup_does_not_promote(self):
        p = pool()
        blocks = p.get_new_blocks(2)
        hs = cache_blocks(p, blocks, 1)
        release(p, blocks)
        for _ in range(10):
            p.get_cached_block(hs[0])
        self.assertEqual(p._retention_queue.stats()["hit_touches"], 0)
        self.assertEqual(len(p._retention_queue.protected), 0)

    def test_hit_on_active_block_and_last_release(self):
        p = pool()
        blocks = p.get_new_blocks(2)
        cache_blocks(p, blocks, 1)
        p.touch(blocks)
        release(p, blocks)
        self.assertTrue(all(b.ref_cnt == 1 for b in blocks))
        self.assertEqual(len(p._retention_queue.protected), 0)
        release(p, blocks)
        self.assertEqual(len(p._retention_queue.protected), 2)
        self.invariant(p)

    def test_tail_first_release_in_both_segments(self):
        p = pool(13)
        blocks = p.get_new_blocks(3)
        cache_blocks(p, blocks, 1)
        release(p, blocks)
        self.assertEqual(list(p._retention_queue.probation)[-3:], [3, 2, 1])
        p.touch(blocks)
        release(p, blocks)
        self.assertEqual(list(p._retention_queue.protected), [3, 2, 1])

    def test_scan_retains_hot_prefix(self):
        hits = {}
        for mode in ("lru", "slru"):
            p = pool(9, mode)
            hot = p.get_new_blocks(3)
            hs = cache_blocks(p, hot, 1)
            release(p, hot)
            p.touch(match(p, hs))
            release(p, hot)
            for key in range(2, 18):
                cold = p.get_new_blocks(2)
                cache_blocks(p, cold, key)
                release(p, cold)
                self.invariant(p)
            hits[mode] = len(match(p, hs))
        self.assertEqual(hits, {"lru": 0, "slru": 3})

    def test_new_hotspot_demotes_old_hotspot(self):
        p = pool(9)
        old = p.get_new_blocks(3)
        old_hs = cache_blocks(p, old, 1)
        release(p, old)
        p.touch(old)
        release(p, old)
        for key in range(2, 9):
            new = p.get_new_blocks(3)
            cache_blocks(p, new, key)
            release(p, new)
            p.touch(new)
            release(p, new)
            self.invariant(p)
        self.assertGreater(p._retention_queue.stats()["demotions"], 0)
        self.assertEqual(len(match(p, old_hs)), 0)

    def test_protected_is_evictable_when_probation_empty(self):
        p = pool(5)
        hot = p.get_new_blocks(2)
        cache_blocks(p, hot, 1)
        release(p, hot)
        p.touch(hot)
        release(p, hot)
        active = p.get_new_blocks(2)
        replacement = p.get_new_blocks(1)
        self.assertEqual(p._retention_queue.stats()["protected_evictions"], 1)
        self.assertTrue(replacement[0].block_hash is None)
        release(p, active + replacement)
        self.invariant(p)

    def test_physical_reassignment_clears_history(self):
        p = pool(5)
        blocks = p.get_new_blocks(4)
        cache_blocks(p, blocks, 1)
        p.touch(blocks)
        release(p, blocks)
        release(p, blocks)
        reused = p.get_new_blocks(4)
        cache_blocks(p, reused, 2)
        release(p, reused)
        self.assertEqual(len(p._retention_queue.protected), 0)
        self.assertEqual(set(p._retention_queue._reuse), {0})
        self.invariant(p)

    def test_successful_reset_clears_hashes_and_policy(self):
        p = pool()
        blocks = p.get_new_blocks(2)
        cache_blocks(p, blocks, 1)
        p.touch(blocks)
        release(p, blocks)
        release(p, blocks)
        self.assertTrue(p.reset_prefix_cache())
        self.assertFalse(p.cached_block_hash_to_block)
        self.assertEqual(p._retention_queue.stats()["hit_touches"], 0)
        self.assertFalse(p._retention_queue.protected)
        self.invariant(p)

    def test_busy_reset_does_not_erase_history(self):
        p = pool()
        blocks = p.get_new_blocks(2)
        hs = cache_blocks(p, blocks, 1)
        p.touch(blocks)
        before = p._retention_queue.stats()
        self.assertFalse(p.reset_prefix_cache())
        self.assertEqual(before, p._retention_queue.stats())
        self.assertIsNotNone(p.get_cached_block(hs[0]))

    def test_duplicate_hash_blocks_and_events(self):
        p = pool(5, events=True)
        a, b = p.get_new_blocks(1), p.get_new_blocks(1)
        hs = cache_blocks(p, a, 1)
        cache_blocks(p, b, 1)
        release(p, a)
        release(p, b)
        p.get_new_blocks(3)  # two never-used blocks, then only one duplicate
        self.assertEqual(len(p.cached_block_hash_to_block[hs[0]]), 1)
        self.assertIs(p.get_cached_block(hs[0]), b[0])
        names = [type(e).__name__ for e in p.take_events()]
        self.assertEqual(names.count("BlockStored"), 2)
        self.assertEqual(names.count("BlockRemoved"), 1)
        self.invariant(p)

    def test_prefix_hole_stops_actual_manager_lookup(self):
        p = pool()
        blocks = p.get_new_blocks(3)
        hs = cache_blocks(p, blocks, 1)
        release(p, blocks)
        # Isolate the manager's actual lookup contract, regardless of policy.
        p._maybe_evict_cached_block(blocks[1])
        self.assertEqual(len(match(p, hs)), 1)
        self.assertIsNotNone(p.get_cached_block(hs[2]))

    def test_invalid_config_and_disabled_caching(self):
        for ratio in ("nan", "inf", "0", "1", "-0.1", "bad"):
            with self.assertRaises(ValueError):
                pool(ratio=ratio)
        with self.assertRaises(ValueError):
            pool(policy="typo")
        p = pool(caching=False)
        self.assertIsInstance(p.free_block_queue, utils.FreeKVCacheBlockQueue)

    def test_tiny_pool(self):
        p = pool(2)
        b = p.get_new_blocks(1)
        cache_blocks(p, b, 1)
        p.touch(b)
        release(p, b)
        release(p, b)
        self.assertEqual(p._retention_queue.protected_limit, 0)
        self.invariant(p)

    def test_randomized_shared_lifecycle(self):
        for seed in range(5):
            rng = random.Random(seed)
            p = pool(33)
            owners = []
            for step in range(1200):
                action = rng.randrange(3)
                if action == 0 and p.get_num_free_blocks():
                    blocks = p.get_new_blocks(rng.randint(1, min(4, p.get_num_free_blocks())))
                    cache_blocks(p, blocks, step + 1)
                    owners.append(blocks)
                elif action == 1 and p.cached_block_hash_to_block:
                    h = rng.choice(list(p.cached_block_hash_to_block))
                    blocks = [p.get_cached_block(h)]
                    p.touch(blocks)
                    owners.append(blocks)
                elif owners:
                    release(p, owners.pop(rng.randrange(len(owners))))
                self.invariant(p)
            for blocks in owners:
                release(p, blocks)
            self.assertTrue(p.reset_prefix_cache())
            self.invariant(p)

    def test_lru_matches_uploaded_original(self):
        for seed in range(3):
            rng = random.Random(seed)
            a, b = pool(25, "lru", cls=original.BlockPool), pool(25, "lru")
            owners = []
            for step in range(800):
                action = rng.randrange(3)
                if action == 0 and a.get_num_free_blocks():
                    n = rng.randint(1, min(4, a.get_num_free_blocks()))
                    aa, bb = a.get_new_blocks(n), b.get_new_blocks(n)
                    self.assertEqual([x.block_id for x in aa], [x.block_id for x in bb])
                    cache_blocks(a, aa, step + 1)
                    cache_blocks(b, bb, step + 1)
                    owners.append((aa, bb))
                elif action == 1 and a.cached_block_hash_to_block:
                    h = rng.choice(list(a.cached_block_hash_to_block))
                    aa, bb = [a.get_cached_block(h)], [b.get_cached_block(h)]
                    a.touch(aa)
                    b.touch(bb)
                    owners.append((aa, bb))
                elif owners:
                    aa, bb = owners.pop(rng.randrange(len(owners)))
                    release(a, aa)
                    release(b, bb)
                self.assertEqual([x.block_id for x in a.free_block_queue.get_all_free_blocks()],
                                 [x.block_id for x in b.free_block_queue.get_all_free_blocks()])
                self.assertEqual([(x.ref_cnt, x.block_hash) for x in a.blocks],
                                 [(x.ref_cnt, x.block_hash) for x in b.blocks])
            for aa, bb in owners:
                release(a, aa)
                release(b, bb)
            self.assertTrue(a.reset_prefix_cache())
            self.assertTrue(b.reset_prefix_cache())


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    print("Mode:", "installed vLLM (CPU logic)" if INSTALLED else "uploaded-source snapshot; no GPU")
    unittest.main(verbosity=2)
