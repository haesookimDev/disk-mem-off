from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LayerCostEstimate:
    """EMA-smoothed cost estimate for a single layer."""

    layer_id: int
    h2d_ms: float = 0.0
    compute_ms: float = 0.0
    sample_count: int = 0


@dataclass(slots=True)
class CostAwareScheduler:
    """Dynamically adjusts prefetch lookahead based on measured layer costs.

    Uses exponential moving average (EMA) of per-layer h2d_ms and compute_ms
    to decide how many future layers' H2D transfers can overlap with the current
    layer's compute time.

    Satisfies the PrefetchScheduler protocol.
    """

    min_lookahead: int = 1
    max_lookahead: int = 4
    ema_alpha: float = 0.3
    _costs: dict[int, LayerCostEstimate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_lookahead < 1:
            raise ValueError("min_lookahead must be >= 1")
        if self.max_lookahead < self.min_lookahead:
            raise ValueError("max_lookahead must be >= min_lookahead")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0.0, 1.0]")

    def feed_metrics(self, layer_metrics: list) -> None:
        """Update cost model with observed LayerMetrics from a completed run."""
        for lm in layer_metrics:
            existing = self._costs.get(lm.layer_id)
            if existing is None:
                self._costs[lm.layer_id] = LayerCostEstimate(
                    layer_id=lm.layer_id,
                    h2d_ms=lm.h2d_ms,
                    compute_ms=lm.compute_ms,
                    sample_count=1,
                )
            else:
                alpha = self.ema_alpha
                existing.h2d_ms = alpha * lm.h2d_ms + (1 - alpha) * existing.h2d_ms
                existing.compute_ms = alpha * lm.compute_ms + (1 - alpha) * existing.compute_ms
                existing.sample_count += 1

    def _effective_lookahead(self, ordered_layer_ids: list[int], current_index: int) -> int:
        """Compute dynamic lookahead based on cost estimates.

        While current layer computes, count how many future H2D transfers
        can fit within the compute budget.
        """
        current_id = ordered_layer_ids[current_index]
        current_cost = self._costs.get(current_id)

        if current_cost is None or current_cost.sample_count == 0:
            return self.min_lookahead

        compute_budget_ms = current_cost.compute_ms
        if compute_budget_ms <= 0:
            return self.max_lookahead

        accumulated_h2d_ms = 0.0
        lookahead = 0

        for future_idx in range(current_index + 1, len(ordered_layer_ids)):
            future_id = ordered_layer_ids[future_idx]
            future_cost = self._costs.get(future_id)
            h2d_est = future_cost.h2d_ms if future_cost else compute_budget_ms
            accumulated_h2d_ms += h2d_est
            lookahead += 1
            if accumulated_h2d_ms >= compute_budget_ms:
                break

        return max(self.min_lookahead, min(lookahead, self.max_lookahead))

    def warmup_prefetch_ids(self, ordered_layer_ids: list[int]) -> list[int]:
        """Prefetch first min_lookahead layers (cold start)."""
        return ordered_layer_ids[: self.min_lookahead]

    def next_prefetch_id(
        self, ordered_layer_ids: list[int], current_index: int
    ) -> int | None:
        lookahead = self._effective_lookahead(ordered_layer_ids, current_index)
        next_index = current_index + lookahead
        if next_index >= len(ordered_layer_ids):
            return None
        return ordered_layer_ids[next_index]
