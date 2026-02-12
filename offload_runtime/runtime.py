from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from offload_runtime.backends.base import DeviceBackend
from offload_runtime.scheduler.lookahead import LookaheadScheduler
from offload_runtime.storage.base import LayerStorage
from offload_runtime.types import DeviceBuffer, LayerSpec


class LayerExecutor(Protocol):
    """Executes one layer using device weights already copied to device."""

    def run_layer(
        self,
        layer: LayerSpec,
        activations: Any,
        device_weights: DeviceBuffer,
        backend: DeviceBackend,
        stream: Any,
    ) -> Any:
        ...


@dataclass(slots=True)
class LayerMetrics:
    layer_id: int
    h2d_ms: float = 0.0
    compute_ms: float = 0.0
    stall_ms: float = 0.0
    nbytes: int = 0


@dataclass(slots=True)
class RuntimeMetrics:
    transferred_bytes: int = 0
    transfer_seconds: float = 0.0
    compute_seconds: float = 0.0
    layer_count: int = 0
    end_to_end_seconds: float = 0.0
    layer_metrics: list[LayerMetrics] = field(default_factory=list)

    @property
    def effective_bandwidth_gbps(self) -> float:
        if self.transfer_seconds <= 0:
            return 0.0
        return (self.transferred_bytes / 1e9) / self.transfer_seconds

    @property
    def overlap_ratio(self) -> float:
        total = self.transfer_seconds + self.compute_seconds
        if total <= 0 or self.end_to_end_seconds <= 0:
            return 0.0
        return 1.0 - (self.end_to_end_seconds / total)


class OffloadRuntime:
    def __init__(
        self,
        layers: list[LayerSpec],
        backend: DeviceBackend,
        storage: LayerStorage,
        scheduler: LookaheadScheduler,
        executor: LayerExecutor,
    ) -> None:
        self.layers = {layer.layer_id: layer for layer in layers}
        self.backend = backend
        self.storage = storage
        self.scheduler = scheduler
        self.executor = executor
        self.transfer_stream = backend.create_stream("transfer")
        self.compute_stream = backend.create_stream("compute")

    def __enter__(self) -> "OffloadRuntime":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self.backend.destroy_stream(self.transfer_stream)
        self.backend.destroy_stream(self.compute_stream)

    def run_inference(self, ordered_layer_ids: list[int], inputs: Any) -> tuple[Any, RuntimeMetrics]:
        unknown = [lid for lid in ordered_layer_ids if lid not in self.layers]
        if unknown:
            raise ValueError(f"Unknown layer IDs: {unknown}")

        metrics = RuntimeMetrics(layer_count=len(ordered_layer_ids))
        activations = inputs
        wall_start = time.perf_counter()

        for layer_id in self.scheduler.warmup_prefetch_ids(ordered_layer_ids):
            self.storage.request(layer_id)

        for index, layer_id in enumerate(ordered_layer_ids):
            layer = self.layers[layer_id]
            lm = LayerMetrics(layer_id=layer_id, nbytes=layer.nbytes)

            t_stall = time.perf_counter()
            self.storage.wait(layer_id)
            host_weights = self.storage.get(layer_id)
            lm.stall_ms = (time.perf_counter() - t_stall) * 1000

            device_weights = self.backend.alloc_device(layer.nbytes)

            t0 = time.perf_counter()
            self.backend.copy_h2d_async(device_weights, host_weights, self.transfer_stream)
            transfer_event = self.backend.record_event(self.transfer_stream)
            self.backend.wait_event(self.compute_stream, transfer_event)
            self.backend.destroy_event(transfer_event)
            h2d_elapsed = time.perf_counter() - t0
            lm.h2d_ms = h2d_elapsed * 1000
            metrics.transfer_seconds += h2d_elapsed
            metrics.transferred_bytes += layer.nbytes

            t1 = time.perf_counter()
            activations = self.executor.run_layer(
                layer=layer,
                activations=activations,
                device_weights=device_weights,
                backend=self.backend,
                stream=self.compute_stream,
            )
            compute_elapsed = time.perf_counter() - t1
            lm.compute_ms = compute_elapsed * 1000
            metrics.compute_seconds += compute_elapsed

            self.backend.free_device(device_weights)
            self.storage.release(layer_id)

            metrics.layer_metrics.append(lm)

            next_layer_id = self.scheduler.next_prefetch_id(ordered_layer_ids, index)
            if next_layer_id is not None:
                self.storage.request(next_layer_id)

        self.backend.synchronize_stream(self.transfer_stream)
        self.backend.synchronize_stream(self.compute_stream)
        metrics.end_to_end_seconds = time.perf_counter() - wall_start
        return activations, metrics
