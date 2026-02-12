from __future__ import annotations

from typing import Protocol


class PrefetchScheduler(Protocol):
    """Protocol that all prefetch schedulers must satisfy."""

    def warmup_prefetch_ids(self, ordered_layer_ids: list[int]) -> list[int]:
        ...

    def next_prefetch_id(
        self, ordered_layer_ids: list[int], current_index: int
    ) -> int | None:
        ...
