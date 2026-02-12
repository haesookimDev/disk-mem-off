from __future__ import annotations

from typing import Protocol

from offload_runtime.types import HostBuffer


class LayerStorage(Protocol):
    """Storage adapter API for layer-wise weight fetching."""

    def request(self, layer_id: int) -> None:
        ...

    def wait(self, layer_id: int) -> None:
        ...

    def get(self, layer_id: int) -> HostBuffer:
        ...

    def release(self, layer_id: int) -> None:
        ...

