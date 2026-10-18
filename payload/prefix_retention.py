# SPDX-License-Identifier: Apache-2.0
"""Bounded, block-level SLRU prototype for the supplied vLLM 0.9.2 sources.

No ghost list, predictor, request reordering, or change to KV tensor storage.
Queue membership contains only evictable blocks. A hit on an active block is
remembered, but the block enters a queue only after its last user releases it.
"""
from collections import OrderedDict
import math
import sys


class SegmentedFreeBlockQueue:
    def __init__(self, blocks, protected_ratio=0.5):
        if not math.isfinite(protected_ratio) or not 0 < protected_ratio < 1:
            raise ValueError("protected_ratio must be finite and between 0 and 1")
        # Block 0 becomes the pool's null block immediately after construction.
        self.protected_limit = int((len(blocks) - 1) * protected_ratio)
        self.probation = OrderedDict((b.block_id, b) for b in blocks)
        self.protected = OrderedDict()
        self._reuse = bytearray(len(blocks))
        self._counters = self._empty_counters()

    @staticmethod
    def _empty_counters():
        return dict(hit_touches=0, protected_admissions=0, demotions=0,
                    probation_evictions=0, protected_evictions=0)

    @property
    def num_free_blocks(self):
        return len(self.probation) + len(self.protected)

    def record_hit(self, block):
        # Called on allocation of an actual cache hit, not speculative lookup.
        # This also covers hits when ref_cnt > 0 (concurrent reuse).
        if block.is_null or block.block_hash is None:
            return
        self._reuse[block.block_id] = 1
        self._counters["hit_touches"] += 1

    def remove(self, block):
        # Removing for a cache hit preserves history across the active interval.
        bid = block.block_id
        queue = self.probation if bid in self.probation else self.protected
        removed = queue.pop(bid)
        assert removed is block

    def popleft(self):
        queue = self.probation if self.probation else self.protected
        if not queue:
            raise ValueError("No free blocks available")
        bid, block = queue.popitem(last=False)
        assert block.ref_cnt == 0 and not block.is_null
        if block.block_hash is not None:
            key = ("probation_evictions" if queue is self.probation
                   else "protected_evictions")
            self._counters[key] += 1
        # Physical block IDs are reused: old content must not inherit priority.
        self._reuse[bid] = 0
        return block

    def append(self, block):
        bid = block.block_id
        assert block.ref_cnt == 0 and not block.is_null
        assert bid not in self.probation and bid not in self.protected
        if block.block_hash is not None and self._reuse[bid]:
            self.protected[bid] = block
            self._counters["protected_admissions"] += 1
            # A release adds one block, so at most one demotion is necessary.
            if len(self.protected) > self.protected_limit:
                old_id, old_block = self.protected.popitem(last=False)
                self._reuse[old_id] = 0
                self.probation[old_id] = old_block
                self._counters["demotions"] += 1
        else:
            self._reuse[bid] = 0
            self.probation[bid] = block

    def get_all_free_blocks(self):
        # Debug / tests only. This is the next-allocation order, not hash order.
        return list(self.probation.values()) + list(self.protected.values())

    def reset_history(self):
        # Called only after BlockPool's idle check and hash invalidation.
        self.probation.update(self.protected)
        self.protected.clear()
        self._reuse[:] = bytes(len(self._reuse))
        self._counters = self._empty_counters()

    def stats(self):
        return {
            **self._counters,
            "probation_blocks": len(self.probation),
            "protected_blocks": len(self.protected),
            "protected_limit": self.protected_limit,
            # Includes containers/bytearray, excludes shared block objects and
            # interpreter allocator fragmentation. This is not process RSS.
            "container_bytes_shallow": sum(map(sys.getsizeof, (
                self.probation, self.protected, self._reuse, self._counters))),
        }
