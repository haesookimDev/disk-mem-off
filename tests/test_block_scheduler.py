from __future__ import annotations

import pytest

from offload_runtime.scheduler.block_scheduler import BlockScheduler


class TestBlockScheduler:
    def test_invalid_block_size(self) -> None:
        with pytest.raises(ValueError, match="block_size must be >= 1"):
            BlockScheduler(block_size=0)

    def test_lookahead_zero_is_valid(self) -> None:
        sched = BlockScheduler(block_size=2, lookahead=0)
        assert sched.lookahead == 0

    def test_lookahead_zero_warmup_empty(self) -> None:
        sched = BlockScheduler(block_size=2, lookahead=0)
        assert sched.warmup_prefetch_ids([0, 1, 2, 3]) == []

    def test_lookahead_zero_next_returns_none(self) -> None:
        sched = BlockScheduler(block_size=2, lookahead=0)
        ids = [0, 1, 2, 3]
        assert sched.next_prefetch_id(ids, 1) is None

    def test_invalid_lookahead_negative(self) -> None:
        with pytest.raises(ValueError, match="lookahead must be >= 0"):
            BlockScheduler(block_size=2, lookahead=-1)

    def test_warmup_single_block(self) -> None:
        sched = BlockScheduler(block_size=2, lookahead=1)
        ids = [0, 1, 2, 3, 4, 5]
        assert sched.warmup_prefetch_ids(ids) == [0, 1]

    def test_warmup_multiple_blocks(self) -> None:
        sched = BlockScheduler(block_size=2, lookahead=2)
        ids = [0, 1, 2, 3, 4, 5]
        assert sched.warmup_prefetch_ids(ids) == [0, 1, 2, 3]

    def test_warmup_exceeds_total(self) -> None:
        sched = BlockScheduler(block_size=3, lookahead=5)
        ids = [0, 1, 2, 3]
        assert sched.warmup_prefetch_ids(ids) == [0, 1, 2, 3]

    def test_next_prefetch_id_triggers_at_block_end(self) -> None:
        sched = BlockScheduler(block_size=2, lookahead=1)
        ids = [0, 1, 2, 3, 4, 5]
        # index 0 is not block end -> None
        assert sched.next_prefetch_id(ids, 0) is None
        # index 1 is block end -> prefetch block 1 (first: 2)
        assert sched.next_prefetch_id(ids, 1) == 2
        # index 2 is not block end -> None
        assert sched.next_prefetch_id(ids, 2) is None
        # index 3 is block end -> prefetch block 2 (first: 4)
        assert sched.next_prefetch_id(ids, 3) == 4

    def test_next_prefetch_id_returns_none_at_end(self) -> None:
        sched = BlockScheduler(block_size=2, lookahead=1)
        ids = [0, 1, 2, 3]
        assert sched.next_prefetch_id(ids, 3) is None

    def test_get_block_prefetch_ids(self) -> None:
        sched = BlockScheduler(block_size=2, lookahead=1)
        ids = [0, 1, 2, 3, 4, 5]
        assert sched.get_block_prefetch_ids(ids, 0) == []
        assert sched.get_block_prefetch_ids(ids, 1) == [2, 3]
        assert sched.get_block_prefetch_ids(ids, 3) == [4, 5]

    def test_odd_layer_count(self) -> None:
        sched = BlockScheduler(block_size=2, lookahead=1)
        ids = [0, 1, 2, 3, 4]
        assert sched.warmup_prefetch_ids(ids) == [0, 1]
        assert sched.get_block_prefetch_ids(ids, 1) == [2, 3]
        assert sched.get_block_prefetch_ids(ids, 3) == [4]

    def test_block_size_1_behaves_like_lookahead(self) -> None:
        sched = BlockScheduler(block_size=1, lookahead=2)
        ids = [0, 1, 2, 3, 4]
        assert sched.warmup_prefetch_ids(ids) == [0, 1]
        assert sched.next_prefetch_id(ids, 0) == 2
        assert sched.next_prefetch_id(ids, 1) == 3
        assert sched.next_prefetch_id(ids, 2) == 4
        assert sched.next_prefetch_id(ids, 3) is None
