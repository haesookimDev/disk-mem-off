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
