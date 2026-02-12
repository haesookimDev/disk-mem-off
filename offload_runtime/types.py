from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LayerSpec:
    layer_id: int
    name: str
    nbytes: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HostBuffer:
    view: memoryview
    pinned: bool = False

    @property
    def nbytes(self) -> int:
        return self.view.nbytes


@dataclass(slots=True)
class DeviceBuffer:
    handle: Any
    nbytes: int
    backend: str

