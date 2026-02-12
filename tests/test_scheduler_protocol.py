from __future__ import annotations

from typing import Any

from offload_runtime.backends.null_backend import NullBackend
from offload_runtime.runtime import OffloadRuntime
from offload_runtime.scheduler.block_scheduler import BlockScheduler
from offload_runtime.scheduler.lookahead import LookaheadScheduler
from offload_runtime.storage.in_memory import InMemoryStorage
from offload_runtime.types import DeviceBuffer, LayerSpec


class SumExecutor:
    def run_layer(self, layer: LayerSpec, activations: Any, device_weights: DeviceBuffer, backend: Any, stream: Any) -> int:
        _ = (device_weights, backend, stream)
        return activations + layer.layer_id


class RecordingExecutor:
    def __init__(self) -> None:
        self.executed: list[int] = []

    def run_layer(self, layer: LayerSpec, activations: Any, device_weights: DeviceBuffer, backend: Any, stream: Any) -> Any:
        _ = (device_weights, backend, stream)
        self.executed.append(layer.layer_id)
        return activations


def _make_layers(count: int, nbytes: int = 16) -> list[LayerSpec]:
    return [LayerSpec(layer_id=i, name=f"L{i}", nbytes=nbytes) for i in range(count)]


def _make_storage(count: int, nbytes: int = 16) -> InMemoryStorage:
    return InMemoryStorage({i: b"\x00" * nbytes for i in range(count)})


class TestSchedulerProtocol:
    def test_runtime_accepts_lookahead_scheduler(self) -> None:
        runtime = OffloadRuntime(
            layers=_make_layers(3),
            backend=NullBackend(),
            storage=_make_storage(3),
            scheduler=LookaheadScheduler(lookahead=1),
            executor=SumExecutor(),
        )
        out, _ = runtime.run_inference([0, 1, 2], inputs=0)
        assert out == 3

    def test_runtime_accepts_block_scheduler(self) -> None:
        runtime = OffloadRuntime(
            layers=_make_layers(6),
            backend=NullBackend(),
            storage=_make_storage(6),
            scheduler=BlockScheduler(block_size=2, lookahead=1),
            executor=SumExecutor(),
        )
        out, metrics = runtime.run_inference([0, 1, 2, 3, 4, 5], inputs=0)
        assert out == 15
        assert metrics.layer_count == 6

    def test_block_scheduler_execution_order(self) -> None:
        executor = RecordingExecutor()
        runtime = OffloadRuntime(
            layers=_make_layers(4),
            backend=NullBackend(),
            storage=_make_storage(4),
            scheduler=BlockScheduler(block_size=2, lookahead=1),
            executor=executor,
        )
        runtime.run_inference([3, 1, 0, 2], inputs=None)
        assert executor.executed == [3, 1, 0, 2]

    def test_block_scheduler_with_buffer_pool(self) -> None:
        backend = NullBackend()
        with OffloadRuntime(
            layers=_make_layers(4),
            backend=backend,
            storage=_make_storage(4),
            scheduler=BlockScheduler(block_size=2, lookahead=1),
            executor=SumExecutor(),
            use_buffer_pool=True,
        ) as runtime:
            out, _ = runtime.run_inference([0, 1, 2, 3], inputs=0)
            assert out == 6
        assert len(backend._buffers) == 0
