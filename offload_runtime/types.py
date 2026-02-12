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


@dataclass(frozen=True, slots=True)
class LoRASpec:
    """Describes a LoRA adapter pair for a layer."""

    layer_id: int
    rank: int
    lora_a_nbytes: int
    lora_b_nbytes: int
    alpha: float = 1.0

