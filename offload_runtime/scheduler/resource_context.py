from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LayerSizeInfo:
    """Size metadata for a single layer."""

    layer_id: int
    nbytes: int
    tensor_count: int = 1


@dataclass(slots=True)
class ResourceSnapshot:
    """Point-in-time system resource snapshot."""

    available_ram_bytes: int = 0
    total_ram_bytes: int = 0
    estimated_disk_bw_bytes_per_sec: float = 500_000_000  # conservative 500 MB/s default


@dataclass(slots=True)
class LayerFeedback:
    """Per-layer timing feedback from the runtime."""

    layer_id: int
    stall_ms: float = 0.0
    disk_read_ms: float = 0.0
    h2d_ms: float = 0.0
    compute_ms: float = 0.0
    nbytes: int = 0


@dataclass(slots=True)
class ResourceContext:
    """Aggregate resource context for scheduler decision-making.

    Populated by the runtime and fed to the scheduler.
    Not part of the PrefetchScheduler Protocol -- only used by schedulers
    that opt in (e.g. ResourceAwareScheduler).
    """

    resources: ResourceSnapshot = field(default_factory=ResourceSnapshot)
    layer_sizes: dict[int, LayerSizeInfo] = field(default_factory=dict)
    layer_feedback: dict[int, LayerFeedback] = field(default_factory=dict)

    # Total bytes currently prefetched and buffered in RAM.
    buffered_bytes: int = 0

    # Fraction of available_ram to use for prefetch buffers.
    ram_budget_fraction: float = 0.5

    @property
    def ram_budget_bytes(self) -> int:
        return int(self.resources.available_ram_bytes * self.ram_budget_fraction)

    @property
    def headroom_bytes(self) -> int:
        """How many more bytes we can prefetch before hitting the RAM budget."""
        return max(0, self.ram_budget_bytes - self.buffered_bytes)


def build_layer_sizes(layers: Any) -> dict[int, LayerSizeInfo]:
    """Build a layer_sizes dict from a list of LayerSpec objects."""
    result: dict[int, LayerSizeInfo] = {}
    for layer in layers:
        tensor_count = len(layer.metadata.get("tensors", [])) if hasattr(layer, "metadata") else 1
        result[layer.layer_id] = LayerSizeInfo(
            layer_id=layer.layer_id,
            nbytes=layer.nbytes,
            tensor_count=max(tensor_count, 1),
        )
    return result
