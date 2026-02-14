from __future__ import annotations

import pytest

from offload_runtime.scheduler.reverse_scheduler import ReverseLookaheadScheduler


class TestReverseLookaheadScheduler:
    def test_lookahead_zero_is_valid(self) -> None:
        sched = ReverseLookaheadScheduler(lookahead=0)
        assert sched.lookahead == 0

    def test_lookahead_zero_warmup_empty(self) -> None:
        sched = ReverseLookaheadScheduler(lookahead=0)
        assert sched.warmup_prefetch_ids([4, 3, 2, 1, 0]) == []

    def test_lookahead_zero_next_returns_none(self) -> None:
        sched = ReverseLookaheadScheduler(lookahead=0)
        assert sched.next_prefetch_id([4, 3, 2, 1, 0], 0) is None

    def test_invalid_lookahead_negative(self) -> None:
        with pytest.raises(ValueError, match="lookahead must be >= 0"):
            ReverseLookaheadScheduler(lookahead=-1)

    def test_warmup_returns_first_w_layers(self) -> None:
        sched = ReverseLookaheadScheduler(lookahead=2)
        # In backward pass, the runtime reverses the list: [4, 3, 2, 1, 0]
        reversed_ids = [4, 3, 2, 1, 0]
        assert sched.warmup_prefetch_ids(reversed_ids) == [4, 3]

    def test_next_prefetch_id_normal(self) -> None:
        sched = ReverseLookaheadScheduler(lookahead=2)
        reversed_ids = [4, 3, 2, 1, 0]
        assert sched.next_prefetch_id(reversed_ids, 0) == 2
        assert sched.next_prefetch_id(reversed_ids, 1) == 1
        assert sched.next_prefetch_id(reversed_ids, 2) == 0

    def test_next_prefetch_id_returns_none_at_end(self) -> None:
        sched = ReverseLookaheadScheduler(lookahead=2)
        reversed_ids = [4, 3, 2, 1, 0]
        assert sched.next_prefetch_id(reversed_ids, 3) is None
        assert sched.next_prefetch_id(reversed_ids, 4) is None

    def test_lookahead_1(self) -> None:
        sched = ReverseLookaheadScheduler(lookahead=1)
        reversed_ids = [2, 1, 0]
        assert sched.warmup_prefetch_ids(reversed_ids) == [2]
        assert sched.next_prefetch_id(reversed_ids, 0) == 1
        assert sched.next_prefetch_id(reversed_ids, 1) == 0
        assert sched.next_prefetch_id(reversed_ids, 2) is None

    def test_empty_list(self) -> None:
        sched = ReverseLookaheadScheduler(lookahead=1)
        assert sched.warmup_prefetch_ids([]) == []
