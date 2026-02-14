from __future__ import annotations

from dataclasses import dataclass, field

from .resource_context import LayerFeedback, ResourceContext


@dataclass(slots=True)
class ResourceAwareScheduler:
    """Scheduler that uses RAM availability, layer sizes, and stall feedback
    to dynamically adjust prefetch depth.

    Key behaviours:
    - Computes how many layers fit in the RAM budget for prefetch.
    - Uses stall_ms / disk_read_ms feedback via EMA to predict I/O time.
    - Supports ``lookahead=0`` (on-demand mode): returns empty warmup and
      ``None`` for next, meaning no prefetching at all.
    - Applies back-pressure: reduces lookahead when RAM headroom is exhausted.

    Satisfies the ``PrefetchScheduler`` protocol
    (``warmup_prefetch_ids`` + ``next_prefetch_id``).
    """

    min_lookahead: int = 0
    max_lookahead: int = 4
    ema_alpha: float = 0.3
    context: ResourceContext = field(default_factory=ResourceContext)

    # Internal EMA state
    _io_cost: dict[int, float] = field(default_factory=dict)
    _compute_cost: dict[int, float] = field(default_factory=dict)
    _sample_count: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_lookahead < 0:
            raise ValueError("min_lookahead must be >= 0")
        if self.max_lookahead < self.min_lookahead:
            raise ValueError("max_lookahead must be >= min_lookahead")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0.0, 1.0]")

    # ------------------------------------------------------------------
    # External methods (not part of the Protocol, called by aware callers)
    # ------------------------------------------------------------------

    def set_context(self, context: ResourceContext) -> None:
        """Update the resource context.  Called by runtime before each inference."""
        self.context = context

    def feed_feedback(self, feedback: list[LayerFeedback]) -> None:
        """Incorporate per-layer timing feedback using EMA."""
        for fb in feedback:
            io_ms = max(fb.stall_ms, fb.disk_read_ms)
            existing_count = self._sample_count.get(fb.layer_id, 0)
            if existing_count == 0:
                self._io_cost[fb.layer_id] = io_ms
                self._compute_cost[fb.layer_id] = fb.compute_ms
            else:
                alpha = self.ema_alpha
                self._io_cost[fb.layer_id] = alpha * io_ms + (1 - alpha) * self._io_cost[fb.layer_id]
                self._compute_cost[fb.layer_id] = alpha * fb.compute_ms + (1 - alpha) * self._compute_cost[fb.layer_id]
            self._sample_count[fb.layer_id] = existing_count + 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_lookahead(self, ordered_layer_ids: list[int], current_index: int) -> int:
        """Compute dynamic lookahead considering:
        1. Compute budget vs I/O cost (overlap potential)
        2. RAM headroom (back-pressure)
        3. min/max bounds
        """
        if not self._sample_count:
            return self.min_lookahead

        current_id = ordered_layer_ids[current_index]
        compute_ms = self._compute_cost.get(current_id, 0.0)

        if compute_ms <= 0.01:
            return self._ram_constrained_lookahead(
                ordered_layer_ids, current_index, self.max_lookahead,
            )

        accumulated_io_ms = 0.0
        lookahead = 0
        for future_idx in range(current_index + 1, len(ordered_layer_ids)):
            future_id = ordered_layer_ids[future_idx]
            io_est = self._io_cost.get(future_id, compute_ms)
            accumulated_io_ms += io_est
            lookahead += 1
            if accumulated_io_ms >= compute_ms:
                break

        lookahead = max(self.min_lookahead, min(lookahead, self.max_lookahead))
        return self._ram_constrained_lookahead(ordered_layer_ids, current_index, lookahead)

    def _ram_constrained_lookahead(
        self, ordered_layer_ids: list[int], current_index: int, target: int,
    ) -> int:
        """Reduce *target* lookahead if prefetching would exceed the RAM budget."""
        headroom = self.context.headroom_bytes
        if headroom <= 0:
            return self.min_lookahead

        accumulated_bytes = 0
        constrained = 0
        for i in range(1, target + 1):
            future_idx = current_index + i
            if future_idx >= len(ordered_layer_ids):
                break
            future_id = ordered_layer_ids[future_idx]
            layer_info = self.context.layer_sizes.get(future_id)
            nbytes = layer_info.nbytes if layer_info else 0
            accumulated_bytes += nbytes
            if accumulated_bytes > headroom:
                break
            constrained += 1

        return max(self.min_lookahead, constrained)

    # ------------------------------------------------------------------
    # PrefetchScheduler protocol
    # ------------------------------------------------------------------

    def warmup_prefetch_ids(self, ordered_layer_ids: list[int]) -> list[int]:
        if not ordered_layer_ids:
            return []

        # In on-demand mode with no prior feedback, skip prefetching entirely.
        if self.min_lookahead == 0 and not self._sample_count:
            return []

        n = max(self.min_lookahead, 1) if self._sample_count else self.min_lookahead
        if n == 0:
            return []

        # Apply RAM budget constraint to warmup too.
        budget = self.context.headroom_bytes
        result: list[int] = []
        accumulated = 0
        for lid in ordered_layer_ids[:n]:
            info = self.context.layer_sizes.get(lid)
            size = info.nbytes if info else 0
            accumulated += size
            if budget > 0 and accumulated > budget and result:
                break
            result.append(lid)
        return result

    def next_prefetch_id(
        self, ordered_layer_ids: list[int], current_index: int,
    ) -> int | None:
        lookahead = self._effective_lookahead(ordered_layer_ids, current_index)
        if lookahead == 0:
            return None
        next_index = current_index + lookahead
        if next_index >= len(ordered_layer_ids):
            return None
        return ordered_layer_ids[next_index]
