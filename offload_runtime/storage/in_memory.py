from __future__ import annotations

from offload_runtime.types import HostBuffer


class InMemoryStorage:
    """Simple storage for unit tests and runtime dry-runs."""

    def __init__(self, layer_bytes: dict[int, bytes]) -> None:
        self._layer_bytes = layer_bytes

    def request(self, layer_id: int) -> None:
        _ = layer_id

    def wait(self, layer_id: int) -> None:
        _ = layer_id

    def get(self, layer_id: int) -> HostBuffer:
        payload = self._layer_bytes[layer_id]
        return HostBuffer(view=memoryview(payload), pinned=False)

    def release(self, layer_id: int) -> None:
        _ = layer_id

