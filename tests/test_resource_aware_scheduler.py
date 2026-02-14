from __future__ import annotations

import pytest

from offload_runtime.scheduler.resource_aware import ResourceAwareScheduler
from offload_runtime.scheduler.resource_context import (
    LayerFeedback,
    LayerSizeInfo,
    ResourceContext,
    ResourceSnapshot,
    build_layer_sizes,
)


def _make_context(
    available_ram: int = 1_000_000_000,
    layer_sizes: dict[int, LayerSizeInfo] | None = None,
    buffered_bytes: int = 0,
    ram_budget_fraction: float = 0.5,
) -> ResourceContext:
    return ResourceContext(
        resources=ResourceSnapshot(available_ram_bytes=available_ram),
        layer_sizes=layer_sizes or {},
        buffered_bytes=buffered_bytes,
        ram_budget_fraction=ram_budget_fraction,
    )


class TestResourceContextBasics:
    def test_headroom_bytes(self) -> None:
        ctx = _make_context(available_ram=1000, ram_budget_fraction=0.5, buffered_bytes=200)
        assert ctx.ram_budget_bytes == 500
        assert ctx.headroom_bytes == 300

    def test_headroom_bytes_clamped_to_zero(self) -> None:
        ctx = _make_context(available_ram=1000, ram_budget_fraction=0.5, buffered_bytes=600)
        assert ctx.headroom_bytes == 0

    def test_build_layer_sizes(self) -> None:
        class FakeLayer:
            def __init__(self, layer_id: int, nbytes: int) -> None:
                self.layer_id = layer_id
                self.nbytes = nbytes
        layers = [FakeLayer(0, 100), FakeLayer(1, 200)]
        result = build_layer_sizes(layers)
        assert len(result) == 2
        assert result[0].nbytes == 100
        assert result[1].nbytes == 200


class TestResourceAwareSchedulerValidation:
    def test_invalid_min_lookahead(self) -> None:
        with pytest.raises(ValueError, match="min_lookahead must be >= 0"):
            ResourceAwareScheduler(min_lookahead=-1)

    def test_invalid_max_lt_min(self) -> None:
        with pytest.raises(ValueError, match="max_lookahead must be >= min_lookahead"):
            ResourceAwareScheduler(min_lookahead=3, max_lookahead=2)

    def test_invalid_ema_alpha_zero(self) -> None:
        with pytest.raises(ValueError, match="ema_alpha"):
            ResourceAwareScheduler(ema_alpha=0.0)

    def test_invalid_ema_alpha_too_high(self) -> None:
        with pytest.raises(ValueError, match="ema_alpha"):
            ResourceAwareScheduler(ema_alpha=1.5)


class TestColdStart:
    """No feedback has been provided yet — scheduler should use min_lookahead."""

    def test_cold_warmup_with_min_zero(self) -> None:
        sched = ResourceAwareScheduler(min_lookahead=0, max_lookahead=4)
        assert sched.warmup_prefetch_ids([0, 1, 2, 3]) == []

    def test_cold_next_with_min_zero(self) -> None:
        sched = ResourceAwareScheduler(min_lookahead=0, max_lookahead=4)
        assert sched.next_prefetch_id([0, 1, 2, 3], 0) is None

    def test_cold_warmup_with_min_nonzero(self) -> None:
        sched = ResourceAwareScheduler(min_lookahead=2, max_lookahead=4)
        assert sched.warmup_prefetch_ids([0, 1, 2, 3]) == [0, 1]

    def test_cold_next_with_min_nonzero(self) -> None:
        sched = ResourceAwareScheduler(min_lookahead=2, max_lookahead=4)
        assert sched.next_prefetch_id([0, 1, 2, 3], 0) == 2

    def test_cold_warmup_empty_list(self) -> None:
        sched = ResourceAwareScheduler(min_lookahead=2, max_lookahead=4)
        assert sched.warmup_prefetch_ids([]) == []


class TestOnDemandMode:
    """min_lookahead=0 is the on-demand mode: no prefetching at all."""

    def test_on_demand_warmup_always_empty(self) -> None:
        sched = ResourceAwareScheduler(min_lookahead=0, max_lookahead=4)
        # Even with feedback, on-demand still requires _effective_lookahead > 0
        assert sched.warmup_prefetch_ids([0, 1, 2]) == []

    def test_on_demand_next_none_without_feedback(self) -> None:
        sched = ResourceAwareScheduler(min_lookahead=0, max_lookahead=4)
        assert sched.next_prefetch_id([0, 1, 2], 0) is None

    def test_on_demand_with_feedback_can_prefetch(self) -> None:
        sched = ResourceAwareScheduler(min_lookahead=0, max_lookahead=4)
        ctx = _make_context(
            available_ram=1_000_000_000,
            layer_sizes={i: LayerSizeInfo(layer_id=i, nbytes=1000) for i in range(5)},
        )
        sched.set_context(ctx)
        # Feed feedback: high compute time, low io → should prefetch ahead
        sched.feed_feedback([
            LayerFeedback(layer_id=i, stall_ms=1.0, disk_read_ms=1.0, compute_ms=10.0)
            for i in range(5)
        ])
        ids = [0, 1, 2, 3, 4]
        result = sched.next_prefetch_id(ids, 0)
        # With 10ms compute and ~1ms io each, several layers fit
        assert result is not None


class TestFeedbackLearning:
    def test_feed_feedback_first_sample(self) -> None:
        sched = ResourceAwareScheduler()
        sched.feed_feedback([
            LayerFeedback(layer_id=0, stall_ms=5.0, disk_read_ms=3.0, compute_ms=10.0),
        ])
        assert sched._sample_count[0] == 1
        # io_cost = max(stall_ms, disk_read_ms) = 5.0
        assert sched._io_cost[0] == pytest.approx(5.0)
        assert sched._compute_cost[0] == pytest.approx(10.0)

    def test_feed_feedback_ema_smoothing(self) -> None:
        sched = ResourceAwareScheduler(ema_alpha=0.5)
        sched.feed_feedback([
            LayerFeedback(layer_id=0, stall_ms=2.0, disk_read_ms=2.0, compute_ms=4.0),
        ])
        sched.feed_feedback([
            LayerFeedback(layer_id=0, stall_ms=6.0, disk_read_ms=6.0, compute_ms=8.0),
        ])
        # EMA: 0.5 * 6.0 + 0.5 * 2.0 = 4.0
        assert sched._io_cost[0] == pytest.approx(4.0)
        # EMA: 0.5 * 8.0 + 0.5 * 4.0 = 6.0
        assert sched._compute_cost[0] == pytest.approx(6.0)
        assert sched._sample_count[0] == 2

    def test_high_stall_increases_lookahead(self) -> None:
        """When I/O is slow relative to compute, need more lookahead."""
        sched = ResourceAwareScheduler(min_lookahead=1, max_lookahead=4)
        ctx = _make_context(
            layer_sizes={i: LayerSizeInfo(layer_id=i, nbytes=1000) for i in range(6)},
        )
        sched.set_context(ctx)
        # High compute, low IO → many layers overlap
        sched.feed_feedback([
            LayerFeedback(layer_id=i, stall_ms=1.0, disk_read_ms=1.0, compute_ms=20.0)
            for i in range(6)
        ])
        ids = [0, 1, 2, 3, 4, 5]
        result = sched.next_prefetch_id(ids, 0)
        assert result is not None
        # Lookahead should be high (capped at max_lookahead=4)
        assert result == 4  # index 0 + lookahead 4


class TestRAMBackpressure:
    def test_ram_limits_lookahead(self) -> None:
        """When RAM headroom is small, reduce lookahead."""
        layer_sizes = {i: LayerSizeInfo(layer_id=i, nbytes=100) for i in range(6)}
        # headroom = 500*0.5 - 0 = 250 bytes → fits 2 layers of 100 bytes each
        ctx = _make_context(
            available_ram=500,
            layer_sizes=layer_sizes,
            ram_budget_fraction=0.5,
        )
        sched = ResourceAwareScheduler(min_lookahead=1, max_lookahead=4)
        sched.set_context(ctx)
        # Feed feedback so scheduler uses dynamic lookahead
        sched.feed_feedback([
            LayerFeedback(layer_id=i, stall_ms=1.0, disk_read_ms=1.0, compute_ms=100.0)
            for i in range(6)
        ])
        ids = [0, 1, 2, 3, 4, 5]
        result = sched.next_prefetch_id(ids, 0)
        # Without RAM limit, would be max_lookahead=4
        # With 250 bytes headroom and 100 bytes/layer, fits only 2 → lookahead=2
        assert result == 2

    def test_zero_headroom_uses_min(self) -> None:
        """When RAM is fully consumed, fall back to min_lookahead."""
        ctx = _make_context(available_ram=0)
        sched = ResourceAwareScheduler(min_lookahead=1, max_lookahead=4)
        sched.set_context(ctx)
        sched.feed_feedback([
            LayerFeedback(layer_id=0, stall_ms=1.0, disk_read_ms=1.0, compute_ms=100.0),
        ])
        ids = [0, 1, 2]
        result = sched.next_prefetch_id(ids, 0)
        assert result == 1  # min_lookahead=1

    def test_warmup_respects_ram_budget(self) -> None:
        layer_sizes = {i: LayerSizeInfo(layer_id=i, nbytes=200) for i in range(5)}
        # headroom = 1000*0.5 - 0 = 500 → fits 2 layers of 200 bytes
        ctx = _make_context(
            available_ram=1000,
            layer_sizes=layer_sizes,
            ram_budget_fraction=0.5,
        )
        sched = ResourceAwareScheduler(min_lookahead=1, max_lookahead=4)
        sched.set_context(ctx)
        # Give feedback so warmup uses at least 1
        sched.feed_feedback([
            LayerFeedback(layer_id=i, stall_ms=1.0, disk_read_ms=1.0, compute_ms=10.0)
            for i in range(5)
        ])
        ids = [0, 1, 2, 3, 4]
        warmup = sched.warmup_prefetch_ids(ids)
        # min_lookahead=1, feedback exists → n=max(1,1)=1 → warmup=[0]
        assert len(warmup) >= 1


class TestSetContext:
    def test_set_context_updates(self) -> None:
        sched = ResourceAwareScheduler()
        ctx = _make_context(available_ram=2000)
        sched.set_context(ctx)
        assert sched.context.resources.available_ram_bytes == 2000


class TestEndOfSequence:
    def test_next_returns_none_at_end(self) -> None:
        sched = ResourceAwareScheduler(min_lookahead=2, max_lookahead=4)
        sched.feed_feedback([
            LayerFeedback(layer_id=0, stall_ms=1.0, disk_read_ms=1.0, compute_ms=10.0),
        ])
        ids = [0, 1, 2]
        assert sched.next_prefetch_id(ids, 2) is None

    def test_next_returns_none_near_end(self) -> None:
        sched = ResourceAwareScheduler(min_lookahead=2, max_lookahead=4)
        sched.feed_feedback([
            LayerFeedback(layer_id=i, stall_ms=1.0, disk_read_ms=1.0, compute_ms=10.0)
            for i in range(3)
        ])
        ids = [0, 1, 2]
        # index 1 + lookahead 2 = 3 >= len(ids)=3 → None
        assert sched.next_prefetch_id(ids, 1) is None


class TestProtocolCompatibility:
    """Verify ResourceAwareScheduler satisfies PrefetchScheduler protocol."""

    def test_has_warmup_method(self) -> None:
        sched = ResourceAwareScheduler()
        assert hasattr(sched, "warmup_prefetch_ids")
        assert callable(sched.warmup_prefetch_ids)

    def test_has_next_method(self) -> None:
        sched = ResourceAwareScheduler()
        assert hasattr(sched, "next_prefetch_id")
        assert callable(sched.next_prefetch_id)

    def test_protocol_structural_typing(self) -> None:
        """Verify method signatures match PrefetchScheduler protocol."""
        import inspect
        from offload_runtime.scheduler.base import PrefetchScheduler

        for name in ("warmup_prefetch_ids", "next_prefetch_id"):
            proto_sig = inspect.signature(getattr(PrefetchScheduler, name))
            impl_sig = inspect.signature(getattr(ResourceAwareScheduler, name))
            assert list(proto_sig.parameters.keys()) == list(impl_sig.parameters.keys())
