from __future__ import annotations

import pytest

from offload_runtime.scheduler.lookahead import LookaheadScheduler


class TestLookaheadScheduler:
    def test_invalid_lookahead_zero(self) -> None:
        with pytest.raises(ValueError, match="lookahead must be >= 1"):
            LookaheadScheduler(lookahead=0)

    def test_invalid_lookahead_negative(self) -> None:
        with pytest.raises(ValueError, match="lookahead must be >= 1"):
            LookaheadScheduler(lookahead=-1)

    def test_warmup_returns_first_w_layers(self) -> None:
        sched = LookaheadScheduler(lookahead=2)
        ids = [10, 20, 30, 40, 50]
        assert sched.warmup_prefetch_ids(ids) == [10, 20]

    def test_warmup_when_lookahead_exceeds_total(self) -> None:
        sched = LookaheadScheduler(lookahead=5)
        ids = [1, 2, 3]
        assert sched.warmup_prefetch_ids(ids) == [1, 2, 3]

    def test_warmup_single_layer(self) -> None:
        sched = LookaheadScheduler(lookahead=1)
        assert sched.warmup_prefetch_ids([42]) == [42]

    def test_warmup_empty_list(self) -> None:
        sched = LookaheadScheduler(lookahead=2)
        assert sched.warmup_prefetch_ids([]) == []

    def test_next_prefetch_id_normal(self) -> None:
        sched = LookaheadScheduler(lookahead=2)
        ids = [0, 1, 2, 3, 4]
        assert sched.next_prefetch_id(ids, 0) == 2
        assert sched.next_prefetch_id(ids, 1) == 3
        assert sched.next_prefetch_id(ids, 2) == 4

    def test_next_prefetch_id_returns_none_at_end(self) -> None:
        sched = LookaheadScheduler(lookahead=2)
        ids = [0, 1, 2, 3, 4]
        assert sched.next_prefetch_id(ids, 3) is None
        assert sched.next_prefetch_id(ids, 4) is None

    def test_next_prefetch_id_lookahead_1(self) -> None:
        sched = LookaheadScheduler(lookahead=1)
        ids = [10, 20, 30]
        assert sched.next_prefetch_id(ids, 0) == 20
        assert sched.next_prefetch_id(ids, 1) == 30
        assert sched.next_prefetch_id(ids, 2) is None
