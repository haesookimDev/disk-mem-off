from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LookaheadScheduler:
    lookahead: int = 1

    def __post_init__(self) -> None:
        if self.lookahead < 1:
            raise ValueError("lookahead must be >= 1")

    def warmup_prefetch_ids(self, ordered_layer_ids: list[int]) -> list[int]:
        return ordered_layer_ids[: self.lookahead]

    def next_prefetch_id(self, ordered_layer_ids: list[int], current_index: int) -> int | None:
        next_index = current_index + self.lookahead
        if next_index >= len(ordered_layer_ids):
            return None
        return ordered_layer_ids[next_index]

