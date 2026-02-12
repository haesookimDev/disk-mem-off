from __future__ import annotations

from typing import Any

import pytest

from offload_runtime.backends.null_backend import NullBackend
from offload_runtime.runtime import LayerMetrics, OffloadRuntime, RuntimeMetrics
from offload_runtime.scheduler.lookahead import LookaheadScheduler
from offload_runtime.storage.in_memory import InMemoryStorage
from offload_runtime.types import DeviceBuffer, LayerSpec


class SumExecutor:
    """Test executor that accumulates layer_id into activations."""

    def run_layer(
        self,
        layer: LayerSpec,
        activations: Any,
        device_weights: DeviceBuffer,
        backend: Any,
        stream: Any,
    ) -> int:
        _ = (device_weights, backend, stream)
        return activations + layer.layer_id


class RecordingExecutor:
    """Test executor that records the order of executed layers."""

    def __init__(self) -> None:
        self.executed: list[int] = []

    def run_layer(
        self,
        layer: LayerSpec,
        activations: Any,
        device_weights: DeviceBuffer,
        backend: Any,
        stream: Any,
    ) -> Any:
        _ = (device_weights, backend, stream)
        self.executed.append(layer.layer_id)
        return activations


def _make_layers(count: int, nbytes: int = 16) -> list[LayerSpec]:
    return [LayerSpec(layer_id=i, name=f"L{i}", nbytes=nbytes) for i in range(count)]


def _make_storage(count: int, nbytes: int = 16) -> InMemoryStorage:
    return InMemoryStorage({i: bytes(range(256))[:nbytes] * (nbytes // min(nbytes, 256) + 1)[:nbytes] for i in range(count)})


def _make_storage_simple(count: int, nbytes: int = 16) -> InMemoryStorage:
    return InMemoryStorage({i: b"\x00" * nbytes for i in range(count)})


class TestOffloadRuntime:
    def test_basic_inference_flow(self) -> None:
        layers = _make_layers(3)
        storage = _make_storage_simple(3)
        runtime = OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
        )
        out, metrics = runtime.run_inference([0, 1, 2], inputs=0)
        assert out == 0 + 1 + 2
        assert metrics.layer_count == 3
        assert metrics.transferred_bytes == 16 * 3

    def test_single_layer(self) -> None:
        layers = _make_layers(1)
        storage = _make_storage_simple(1)
        runtime = OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
        )
        out, metrics = runtime.run_inference([0], inputs=100)
        assert out == 100
        assert metrics.layer_count == 1

    def test_execution_order(self) -> None:
        layers = _make_layers(5)
        storage = _make_storage_simple(5)
        executor = RecordingExecutor()
        runtime = OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=2),
            executor=executor,
        )
        runtime.run_inference([4, 2, 0, 3, 1], inputs=None)
        assert executor.executed == [4, 2, 0, 3, 1]

    def test_metrics_transfer_seconds_positive(self) -> None:
        layers = _make_layers(2)
        storage = _make_storage_simple(2)
        runtime = OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
        )
        _, metrics = runtime.run_inference([0, 1], inputs=0)
        assert metrics.transfer_seconds >= 0.0
        assert metrics.compute_seconds >= 0.0

    def test_invalid_layer_id_raises(self) -> None:
        layers = _make_layers(2)
        storage = _make_storage_simple(2)
        runtime = OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
        )
        with pytest.raises(ValueError, match="Unknown layer IDs"):
            runtime.run_inference([0, 1, 99], inputs=0)

    def test_different_lookahead_values(self) -> None:
        for lookahead in [1, 2, 3, 5]:
            layers = _make_layers(5)
            storage = _make_storage_simple(5)
            runtime = OffloadRuntime(
                layers=layers,
                backend=NullBackend(),
                storage=storage,
                scheduler=LookaheadScheduler(lookahead=lookahead),
                executor=SumExecutor(),
            )
            out, _ = runtime.run_inference([0, 1, 2, 3, 4], inputs=0)
            assert out == 10

    def test_device_buffers_freed_after_inference(self) -> None:
        backend = NullBackend()
        layers = _make_layers(3)
        storage = _make_storage_simple(3)
        runtime = OffloadRuntime(
            layers=layers,
            backend=backend,
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
        )
        runtime.run_inference([0, 1, 2], inputs=0)
        assert len(backend._buffers) == 0

    def test_context_manager(self) -> None:
        layers = _make_layers(2)
        storage = _make_storage_simple(2)
        with OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
        ) as runtime:
            out, _ = runtime.run_inference([0, 1], inputs=0)
            assert out == 1

    def test_per_layer_metrics(self) -> None:
        layers = _make_layers(3, nbytes=32)
        storage = _make_storage_simple(3, nbytes=32)
        runtime = OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
        )
        _, metrics = runtime.run_inference([0, 1, 2], inputs=0)

        assert len(metrics.layer_metrics) == 3
        for i, lm in enumerate(metrics.layer_metrics):
            assert lm.layer_id == i
            assert lm.nbytes == 32
            assert lm.h2d_ms >= 0.0
            assert lm.compute_ms >= 0.0
            assert lm.stall_ms >= 0.0

    def test_end_to_end_seconds(self) -> None:
        layers = _make_layers(2)
        storage = _make_storage_simple(2)
        runtime = OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
        )
        _, metrics = runtime.run_inference([0, 1], inputs=0)
        assert metrics.end_to_end_seconds > 0.0

    def test_effective_bandwidth(self) -> None:
        metrics = RuntimeMetrics(
            transferred_bytes=1_000_000_000,
            transfer_seconds=1.0,
        )
        assert metrics.effective_bandwidth_gbps == pytest.approx(1.0)

    def test_effective_bandwidth_zero_time(self) -> None:
        metrics = RuntimeMetrics(transferred_bytes=100, transfer_seconds=0.0)
        assert metrics.effective_bandwidth_gbps == 0.0

    def test_overlap_ratio_no_overlap(self) -> None:
        metrics = RuntimeMetrics(
            transfer_seconds=1.0,
            compute_seconds=1.0,
            end_to_end_seconds=2.0,
        )
        assert metrics.overlap_ratio == pytest.approx(0.0)

    def test_overlap_ratio_full_overlap(self) -> None:
        metrics = RuntimeMetrics(
            transfer_seconds=1.0,
            compute_seconds=1.0,
            end_to_end_seconds=1.0,
        )
        assert metrics.overlap_ratio == pytest.approx(0.5)

    def test_buffer_pool_reuses_buffers(self) -> None:
        backend = NullBackend()
        layers = _make_layers(3)
        storage = _make_storage_simple(3)
        with OffloadRuntime(
            layers=layers,
            backend=backend,
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
            use_buffer_pool=True,
        ) as runtime:
            out, metrics = runtime.run_inference([0, 1, 2], inputs=0)
            assert out == 3
            assert metrics.layer_count == 3
        # After close(), pool drained, backend should be empty
        assert len(backend._buffers) == 0

    def test_pinned_memory_alloc_free(self) -> None:
        backend = NullBackend()
        assert backend.supports_pinned_host is True
        buf = backend.alloc_pinned_host(64)
        assert buf.pinned is True
        assert buf.nbytes == 64
        backend.free_pinned_host(buf)

    def test_pinned_staging_inference(self) -> None:
        layers = _make_layers(3)
        storage = _make_storage_simple(3)
        with OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
            use_pinned_staging=True,
        ) as runtime:
            out, metrics = runtime.run_inference([0, 1, 2], inputs=0)
            assert out == 3
            assert metrics.layer_count == 3
            assert metrics.transferred_bytes == 48

    def test_pinned_staging_with_buffer_pool(self) -> None:
        backend = NullBackend()
        with OffloadRuntime(
            layers=_make_layers(3),
            backend=backend,
            storage=_make_storage_simple(3),
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
            use_buffer_pool=True,
            use_pinned_staging=True,
        ) as runtime:
            out, _ = runtime.run_inference([0, 1, 2], inputs=0)
            assert out == 3
        assert len(backend._buffers) == 0

    def test_pinned_staging_skipped_when_unsupported(self) -> None:
        from offload_runtime.backends.base import DeviceBackend
        from offload_runtime.types import HostBuffer

        class NoPinnedBackend(NullBackend):
            @property
            def supports_pinned_host(self) -> bool:
                return False

        with OffloadRuntime(
            layers=_make_layers(2),
            backend=NoPinnedBackend(),
            storage=_make_storage_simple(2),
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
            use_pinned_staging=True,  # should degrade gracefully
        ) as runtime:
            out, _ = runtime.run_inference([0, 1], inputs=0)
            assert out == 1

    def test_quantized_transfer_int8(self) -> None:
        import struct
        from offload_runtime.quantize import Int8Dequantizer

        # 4 bytes of int8 data per layer -> 16 bytes float32 after dequant
        layers = [
            LayerSpec(layer_id=0, name="L0", nbytes=4, metadata={"dtype": "int8", "scale": 1.0}),
            LayerSpec(layer_id=1, name="L1", nbytes=4, metadata={"dtype": "int8", "scale": 1.0}),
        ]
        storage = InMemoryStorage({
            0: struct.pack("4b", 1, 2, 3, 4),
            1: struct.pack("4b", 5, 6, 7, 8),
        })
        runtime = OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
            dequantizer=Int8Dequantizer(),
        )
        out, metrics = runtime.run_inference([0, 1], inputs=0)
        assert out == 0 + 1  # SumExecutor adds layer_id
        assert metrics.layer_count == 2
        # transferred_bytes = decompressed size (16 bytes * 2 layers)
        assert metrics.transferred_bytes == 32
        # compressed_bytes = original int8 size (4 bytes * 2 layers)
        assert metrics.compressed_bytes == 8
        # per-layer metrics
        assert metrics.layer_metrics[0].compressed_nbytes == 4
        assert metrics.layer_metrics[0].nbytes == 16
        assert metrics.layer_metrics[1].compressed_nbytes == 4

    def test_quantized_transfer_no_dequantizer(self) -> None:
        # Without dequantizer, metadata is ignored
        layers = [
            LayerSpec(layer_id=0, name="L0", nbytes=16, metadata={"dtype": "int8", "scale": 1.0}),
        ]
        storage = _make_storage_simple(1)
        runtime = OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
        )
        out, metrics = runtime.run_inference([0], inputs=0)
        assert out == 0
        assert metrics.transferred_bytes == 16
        assert metrics.compressed_bytes == 0

    def test_quantized_mixed_layers(self) -> None:
        import struct
        from offload_runtime.quantize import CompositeDequantizer

        layers = [
            LayerSpec(layer_id=0, name="L0", nbytes=16),  # non-quantized
            LayerSpec(layer_id=1, name="L1", nbytes=4, metadata={"dtype": "int8", "scale": 1.0}),
        ]
        storage = InMemoryStorage({
            0: b"\x00" * 16,
            1: struct.pack("4b", 1, 2, 3, 4),
        })
        runtime = OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
            dequantizer=CompositeDequantizer(),
        )
        out, metrics = runtime.run_inference([0, 1], inputs=0)
        assert out == 0 + 1
        # Layer 0: 16 bytes (not quantized), Layer 1: 16 bytes (decompressed from 4)
        assert metrics.transferred_bytes == 32
        assert metrics.compressed_bytes == 4  # only layer 1
        assert metrics.layer_metrics[0].compressed_nbytes == 0
        assert metrics.layer_metrics[1].compressed_nbytes == 4
