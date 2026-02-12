from __future__ import annotations

import time
from dataclasses import dataclass
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
class RuntimeMetrics:
    transferred_bytes: int = 0
    transfer_seconds: float = 0.0
    compute_seconds: float = 0.0
    layer_count: int = 0


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
        metrics = RuntimeMetrics(layer_count=len(ordered_layer_ids))
        activations = inputs

        for layer_id in self.scheduler.warmup_prefetch_ids(ordered_layer_ids):
            self.storage.request(layer_id)

        for index, layer_id in enumerate(ordered_layer_ids):
            layer = self.layers[layer_id]

            self.storage.wait(layer_id)
            host_weights = self.storage.get(layer_id)

            device_weights = self.backend.alloc_device(layer.nbytes)

            t0 = time.perf_counter()
            self.backend.copy_h2d_async(device_weights, host_weights, self.transfer_stream)
            transfer_event = self.backend.record_event(self.transfer_stream)
            self.backend.wait_event(self.compute_stream, transfer_event)
            metrics.transfer_seconds += time.perf_counter() - t0
            metrics.transferred_bytes += layer.nbytes

            t1 = time.perf_counter()
            activations = self.executor.run_layer(
                layer=layer,
                activations=activations,
                device_weights=device_weights,
                backend=self.backend,
                stream=self.compute_stream,
            )
            metrics.compute_seconds += time.perf_counter() - t1

            self.backend.free_device(device_weights)
            self.storage.release(layer_id)

            next_layer_id = self.scheduler.next_prefetch_id(ordered_layer_ids, index)
            if next_layer_id is not None:
                self.storage.request(next_layer_id)

        self.backend.synchronize_stream(self.transfer_stream)
        self.backend.synchronize_stream(self.compute_stream)
        return activations, metrics

