from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BlockScheduler:
    """Groups layers into blocks for batch prefetching.

    Instead of prefetching one layer at a time, groups consecutive layers
    into blocks of `block_size` and prefetches `lookahead` blocks ahead.
    """

    block_size: int = 2
    lookahead: int = 1

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.lookahead < 0:
            raise ValueError("lookahead must be >= 0")

    def _blocks(self, ordered_layer_ids: list[int]) -> list[list[int]]:
        return [
            ordered_layer_ids[i : i + self.block_size]
            for i in range(0, len(ordered_layer_ids), self.block_size)
        ]

    def warmup_prefetch_ids(self, ordered_layer_ids: list[int]) -> list[int]:
        blocks = self._blocks(ordered_layer_ids)
        result: list[int] = []
        for block in blocks[: self.lookahead]:
            result.extend(block)
        return result

    def next_prefetch_id(self, ordered_layer_ids: list[int], current_index: int) -> int | None:
        if self.lookahead == 0:
            return None
        blocks = self._blocks(ordered_layer_ids)
        current_block_idx = current_index // self.block_size
        position_in_block = current_index % self.block_size

        if position_in_block != self.block_size - 1 and current_index != len(ordered_layer_ids) - 1:
            return None

        target_block_idx = current_block_idx + self.lookahead
        if target_block_idx >= len(blocks):
            return None

        return blocks[target_block_idx][0]

    def get_block_prefetch_ids(self, ordered_layer_ids: list[int], current_index: int) -> list[int]:
        if self.lookahead == 0:
            return []
        blocks = self._blocks(ordered_layer_ids)
        current_block_idx = current_index // self.block_size
        position_in_block = current_index % self.block_size

        if position_in_block != self.block_size - 1 and current_index != len(ordered_layer_ids) - 1:
            return []

        target_block_idx = current_block_idx + self.lookahead
        if target_block_idx >= len(blocks):
            return []

        return blocks[target_block_idx]
