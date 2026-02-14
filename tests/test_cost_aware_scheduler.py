from __future__ import annotations

import pytest

from offload_runtime.runtime import LayerMetrics
from offload_runtime.scheduler.cost_aware import CostAwareScheduler


class TestCostAwareScheduler:
    def test_cold_start_uses_min_lookahead(self) -> None:
        sched = CostAwareScheduler(min_lookahead=2, max_lookahead=5)
        ids = [0, 1, 2, 3, 4]
        assert sched.warmup_prefetch_ids(ids) == [0, 1]
        assert sched.next_prefetch_id(ids, 0) == 2  # min_lookahead=2

    def test_cold_start_returns_none_at_end(self) -> None:
        sched = CostAwareScheduler(min_lookahead=2, max_lookahead=5)
        ids = [0, 1, 2]
        assert sched.next_prefetch_id(ids, 2) is None

    def test_feed_metrics_creates_cost(self) -> None:
        sched = CostAwareScheduler()
        metrics = [LayerMetrics(layer_id=0, h2d_ms=1.0, compute_ms=5.0)]
        sched.feed_metrics(metrics)
        assert 0 in sched._costs
        assert sched._costs[0].h2d_ms == 1.0
        assert sched._costs[0].compute_ms == 5.0
        assert sched._costs[0].sample_count == 1

    def test_ema_smoothing(self) -> None:
        sched = CostAwareScheduler(ema_alpha=0.5)
        sched.feed_metrics([LayerMetrics(layer_id=0, h2d_ms=2.0, compute_ms=4.0)])
        sched.feed_metrics([LayerMetrics(layer_id=0, h2d_ms=4.0, compute_ms=4.0)])
        # EMA: 0.5 * 4.0 + 0.5 * 2.0 = 3.0
        assert sched._costs[0].h2d_ms == pytest.approx(3.0)
        assert sched._costs[0].sample_count == 2

    def test_high_h2d_increases_lookahead(self) -> None:
        sched = CostAwareScheduler(min_lookahead=1, max_lookahead=5)
        # Current layer has large compute budget, small h2d → more layers fit
        sched.feed_metrics([
            LayerMetrics(layer_id=i, h2d_ms=1.0, compute_ms=10.0)
            for i in range(6)
        ])
        ids = [0, 1, 2, 3, 4, 5]
        # With 10ms compute and 1ms h2d each, ~10 layers fit, capped by max_lookahead=5
        result = sched.next_prefetch_id(ids, 0)
        assert result == 5  # index 0 + lookahead 5

    def test_high_compute_reduces_effective_lookahead(self) -> None:
        sched = CostAwareScheduler(min_lookahead=1, max_lookahead=5)
        # Each h2d takes 10ms, compute takes 1ms → only 1 layer fits
        sched.feed_metrics([
            LayerMetrics(layer_id=i, h2d_ms=10.0, compute_ms=1.0)
            for i in range(6)
        ])
        ids = [0, 1, 2, 3, 4, 5]
        result = sched.next_prefetch_id(ids, 0)
        assert result == 1  # min_lookahead=1

    def test_zero_compute_uses_max_lookahead(self) -> None:
        sched = CostAwareScheduler(min_lookahead=1, max_lookahead=3)
        sched.feed_metrics([
            LayerMetrics(layer_id=i, h2d_ms=1.0, compute_ms=0.0)
            for i in range(5)
        ])
        ids = [0, 1, 2, 3, 4]
        assert sched.next_prefetch_id(ids, 0) == 3  # max_lookahead=3

    def test_min_lookahead_zero_is_valid(self) -> None:
        sched = CostAwareScheduler(min_lookahead=0, max_lookahead=3)
        assert sched.min_lookahead == 0

    def test_min_lookahead_zero_warmup_empty(self) -> None:
        sched = CostAwareScheduler(min_lookahead=0, max_lookahead=3)
        assert sched.warmup_prefetch_ids([0, 1, 2]) == []

    def test_min_lookahead_zero_cold_returns_none(self) -> None:
        sched = CostAwareScheduler(min_lookahead=0, max_lookahead=3)
        assert sched.next_prefetch_id([0, 1, 2], 0) is None

    def test_validation_min_lookahead_negative(self) -> None:
        with pytest.raises(ValueError, match="min_lookahead must be >= 0"):
            CostAwareScheduler(min_lookahead=-1)

    def test_validation_max_lt_min(self) -> None:
        with pytest.raises(ValueError, match="max_lookahead must be >= min_lookahead"):
            CostAwareScheduler(min_lookahead=3, max_lookahead=2)

    def test_validation_ema_alpha(self) -> None:
        with pytest.raises(ValueError, match="ema_alpha"):
            CostAwareScheduler(ema_alpha=0.0)
        with pytest.raises(ValueError, match="ema_alpha"):
            CostAwareScheduler(ema_alpha=1.5)

    def test_warmup_empty_list(self) -> None:
        sched = CostAwareScheduler()
        assert sched.warmup_prefetch_ids([]) == []

    def test_next_prefetch_last_index(self) -> None:
        sched = CostAwareScheduler(min_lookahead=1)
        ids = [0, 1, 2]
        assert sched.next_prefetch_id(ids, 2) is None

    def test_feed_metrics_captures_io_ms(self) -> None:
        sched = CostAwareScheduler()
        metrics = [LayerMetrics(layer_id=0, h2d_ms=1.0, compute_ms=5.0, disk_read_ms=3.0)]
        sched.feed_metrics(metrics)
        assert sched._costs[0].io_ms == pytest.approx(3.0)

    def test_io_ms_uses_stall_as_fallback(self) -> None:
        sched = CostAwareScheduler()
        # disk_read_ms=0.0 → falls back to stall_ms
        metrics = [LayerMetrics(layer_id=0, h2d_ms=1.0, compute_ms=5.0, stall_ms=4.0)]
        sched.feed_metrics(metrics)
        assert sched._costs[0].io_ms == pytest.approx(4.0)

    def test_high_io_reduces_effective_lookahead(self) -> None:
        """When disk I/O is the bottleneck (io_ms >> h2d_ms), prefetch cost increases."""
        sched = CostAwareScheduler(min_lookahead=1, max_lookahead=5)
        # io_ms=10ms dominates h2d_ms=1ms, so prefetch_ms = max(1, 10) = 10ms
        # With 10ms compute budget and 10ms per prefetch, only 1 layer fits
        sched.feed_metrics([
            LayerMetrics(layer_id=i, h2d_ms=1.0, compute_ms=10.0, disk_read_ms=10.0)
            for i in range(6)
        ])
        ids = [0, 1, 2, 3, 4, 5]
        result = sched.next_prefetch_id(ids, 0)
        assert result == 1  # min_lookahead=1

    def test_io_ms_ema_smoothing(self) -> None:
        sched = CostAwareScheduler(ema_alpha=0.5)
        sched.feed_metrics([LayerMetrics(layer_id=0, h2d_ms=1.0, compute_ms=5.0, disk_read_ms=2.0)])
        sched.feed_metrics([LayerMetrics(layer_id=0, h2d_ms=1.0, compute_ms=5.0, disk_read_ms=6.0)])
        # EMA: 0.5 * 6.0 + 0.5 * 2.0 = 4.0
        assert sched._costs[0].io_ms == pytest.approx(4.0)
