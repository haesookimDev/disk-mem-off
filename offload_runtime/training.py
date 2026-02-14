from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from offload_runtime.backends.base import DeviceBackend
from offload_runtime.buffer_pool import DeviceBufferPool
from offload_runtime.pinned_pool import PinnedHostBufferPool
from offload_runtime.scheduler.base import PrefetchScheduler
from offload_runtime.storage.base import LayerStorage
from offload_runtime.types import DeviceBuffer, HostBuffer, LayerSpec, LoRASpec


class TrainingExecutor(Protocol):
    """Executes forward + backward for one layer with LoRA adapters."""

    def forward_layer(
        self,
        layer: LayerSpec,
        lora: LoRASpec | None,
        activations: Any,
        device_weights: DeviceBuffer,
        lora_a: DeviceBuffer | None,
        lora_b: DeviceBuffer | None,
        backend: DeviceBackend,
        stream: Any,
    ) -> tuple[Any, Any]:
        """Returns (output_activations, saved_state_for_backward)."""
        ...

    def backward_layer(
        self,
        layer: LayerSpec,
        lora: LoRASpec | None,
        grad_output: Any,
        saved_state: Any,
        device_weights: DeviceBuffer,
        lora_a: DeviceBuffer | None,
        lora_b: DeviceBuffer | None,
        backend: DeviceBackend,
        stream: Any,
    ) -> tuple[Any, Any, Any]:
        """Returns (grad_input, grad_lora_a, grad_lora_b)."""
        ...


@dataclass(slots=True)
class TrainingLayerMetrics:
    layer_id: int
    forward_h2d_ms: float = 0.0
    forward_compute_ms: float = 0.0
    backward_h2d_ms: float = 0.0
    backward_compute_ms: float = 0.0
    stall_ms: float = 0.0
    disk_read_ms: float = 0.0
    nbytes: int = 0


@dataclass(slots=True)
class TrainingMetrics:
    transferred_bytes: int = 0
    transfer_seconds: float = 0.0
    forward_compute_seconds: float = 0.0
    backward_compute_seconds: float = 0.0
    layer_count: int = 0
    end_to_end_seconds: float = 0.0
    layer_metrics: list[TrainingLayerMetrics] = field(default_factory=list)


class TrainingRuntime:
    """Extends the offload pattern for LoRA fine-tuning.

    Architecture:
    - Forward pass: offloaded base weights + resident LoRA adapters
    - Backward pass: reverse-order re-prefetch of base weights
    - Three streams: transfer, forward_compute, backward_compute
    - LoRA adapter buffers: allocated on device at init, persist across steps
    """

    def __init__(
        self,
        layers: list[LayerSpec],
        lora_specs: dict[int, LoRASpec],
        backend: DeviceBackend,
        storage: LayerStorage,
        forward_scheduler: PrefetchScheduler,
        backward_scheduler: PrefetchScheduler,
        executor: TrainingExecutor,
        use_buffer_pool: bool = False,
        use_pinned_staging: bool = False,
    ) -> None:
        self.layers = {layer.layer_id: layer for layer in layers}
        self.lora_specs = lora_specs
        self.backend = backend
        self.storage = storage
        self.forward_scheduler = forward_scheduler
        self.backward_scheduler = backward_scheduler
        self.executor = executor

        self.transfer_stream = backend.create_stream("transfer")
        self.forward_stream = backend.create_stream("forward_compute")
        self.backward_stream = backend.create_stream("backward_compute")

        self._pool = DeviceBufferPool(backend) if use_buffer_pool else None
        self._pinned_pool: PinnedHostBufferPool | None = None
        if use_pinned_staging and backend.supports_pinned_host:
            self._pinned_pool = PinnedHostBufferPool(backend)

        # Allocate persistent LoRA buffers on device
        self._lora_buffers: dict[int, tuple[DeviceBuffer, DeviceBuffer]] = {}
        for lid, lora in lora_specs.items():
            a = backend.alloc_device(lora.lora_a_nbytes)
            b = backend.alloc_device(lora.lora_b_nbytes)
            self._lora_buffers[lid] = (a, b)

    def __enter__(self) -> TrainingRuntime:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        # Free LoRA buffers
        for a, b in self._lora_buffers.values():
            self.backend.free_device(a)
            self.backend.free_device(b)
        self._lora_buffers.clear()

        if self._pinned_pool is not None:
            self._pinned_pool.drain()
        if self._pool is not None:
            self._pool.drain()

        self.backend.destroy_stream(self.transfer_stream)
        self.backend.destroy_stream(self.forward_stream)
        self.backend.destroy_stream(self.backward_stream)

    def _alloc_device(self, nbytes: int) -> DeviceBuffer:
        if self._pool is not None:
            return self._pool.acquire(nbytes)
        return self.backend.alloc_device(nbytes)

    def _free_device(self, buf: DeviceBuffer) -> None:
        if self._pool is not None:
            self._pool.release(buf)
        else:
            self.backend.free_device(buf)

    def _stage_pinned(self, host_weights: HostBuffer) -> HostBuffer:
        if self._pinned_pool is None or host_weights.pinned:
            return host_weights
        pinned = self._pinned_pool.acquire(host_weights.nbytes)
        pinned.view[: host_weights.nbytes] = host_weights.view[: host_weights.nbytes]
        return pinned

    def _load_and_transfer(
        self, layer_id: int, compute_stream: Any, metrics: TrainingMetrics, lm: TrainingLayerMetrics, phase: str,
    ) -> DeviceBuffer:
        """Wait for storage, H2D copy, return device buffer."""
        t_stall = time.perf_counter()
        self.storage.wait(layer_id)
        host_weights = self.storage.get(layer_id)
        lm.stall_ms += (time.perf_counter() - t_stall) * 1000

        if hasattr(self.storage, "get_disk_read_ms"):
            lm.disk_read_ms += self.storage.get_disk_read_ms(layer_id)

        staged = self._stage_pinned(host_weights)
        actual_nbytes = host_weights.nbytes
        device_weights = self._alloc_device(actual_nbytes)

        t0 = time.perf_counter()
        self.backend.copy_h2d_async(device_weights, staged, self.transfer_stream)
        transfer_event = self.backend.record_event(self.transfer_stream)
        self.backend.wait_event(compute_stream, transfer_event)
        self.backend.destroy_event(transfer_event)
        h2d_elapsed = time.perf_counter() - t0

        if phase == "forward":
            lm.forward_h2d_ms = h2d_elapsed * 1000
        else:
            lm.backward_h2d_ms = h2d_elapsed * 1000

        lm.nbytes = actual_nbytes
        metrics.transfer_seconds += h2d_elapsed
        metrics.transferred_bytes += actual_nbytes

        self.storage.release(layer_id)
        if staged is not host_weights and self._pinned_pool is not None:
            self._pinned_pool.release(staged)

        return device_weights

    def run_training_step(
        self,
        ordered_layer_ids: list[int],
        inputs: Any,
    ) -> tuple[Any, TrainingMetrics]:
        """Execute forward + backward pass with weight offloading."""
        unknown = [lid for lid in ordered_layer_ids if lid not in self.layers]
        if unknown:
            raise ValueError(f"Unknown layer IDs: {unknown}")

        metrics = TrainingMetrics(layer_count=len(ordered_layer_ids))
        wall_start = time.perf_counter()

        # -- Forward pass --
        saved_states: dict[int, Any] = {}
        layer_metrics_map: dict[int, TrainingLayerMetrics] = {}
        activations = inputs

        for layer_id in self.forward_scheduler.warmup_prefetch_ids(ordered_layer_ids):
            self.storage.request(layer_id)

        device_weights: DeviceBuffer | None = None
        try:
            for index, layer_id in enumerate(ordered_layer_ids):
                layer = self.layers[layer_id]
                lm = TrainingLayerMetrics(layer_id=layer_id)

                device_weights = self._load_and_transfer(
                    layer_id, self.forward_stream, metrics, lm, "forward",
                )

                lora = self.lora_specs.get(layer_id)
                lora_a, lora_b = self._lora_buffers.get(layer_id, (None, None))

                t1 = time.perf_counter()
                activations, saved = self.executor.forward_layer(
                    layer=layer,
                    lora=lora,
                    activations=activations,
                    device_weights=device_weights,
                    lora_a=lora_a,
                    lora_b=lora_b,
                    backend=self.backend,
                    stream=self.forward_stream,
                )
                fwd_elapsed = time.perf_counter() - t1
                lm.forward_compute_ms = fwd_elapsed * 1000
                metrics.forward_compute_seconds += fwd_elapsed

                saved_states[layer_id] = saved
                self._free_device(device_weights)
                device_weights = None
                layer_metrics_map[layer_id] = lm

                next_id = self.forward_scheduler.next_prefetch_id(ordered_layer_ids, index)
                if next_id is not None:
                    self.storage.request(next_id)

            # -- Backward pass (reverse order) --
            reversed_ids = list(reversed(ordered_layer_ids))
            grad = activations

            for layer_id in self.backward_scheduler.warmup_prefetch_ids(reversed_ids):
                self.storage.request(layer_id)

            for index, layer_id in enumerate(reversed_ids):
                layer = self.layers[layer_id]
                lm = layer_metrics_map[layer_id]

                device_weights = self._load_and_transfer(
                    layer_id, self.backward_stream, metrics, lm, "backward",
                )

                lora = self.lora_specs.get(layer_id)
                lora_a, lora_b = self._lora_buffers.get(layer_id, (None, None))

                t2 = time.perf_counter()
                grad, grad_a, grad_b = self.executor.backward_layer(
                    layer=layer,
                    lora=lora,
                    grad_output=grad,
                    saved_state=saved_states[layer_id],
                    device_weights=device_weights,
                    lora_a=lora_a,
                    lora_b=lora_b,
                    backend=self.backend,
                    stream=self.backward_stream,
                )
                bwd_elapsed = time.perf_counter() - t2
                lm.backward_compute_ms = bwd_elapsed * 1000
                metrics.backward_compute_seconds += bwd_elapsed

                self._free_device(device_weights)
                device_weights = None

                next_id = self.backward_scheduler.next_prefetch_id(reversed_ids, index)
                if next_id is not None:
                    self.storage.request(next_id)
        finally:
            if device_weights is not None:
                self._free_device(device_weights)
            self.backend.synchronize_stream(self.transfer_stream)
            self.backend.synchronize_stream(self.forward_stream)
            self.backend.synchronize_stream(self.backward_stream)

        metrics.end_to_end_seconds = time.perf_counter() - wall_start
        metrics.layer_metrics = [layer_metrics_map[lid] for lid in ordered_layer_ids]
        return grad, metrics
