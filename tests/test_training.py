from __future__ import annotations

from typing import Any

import pytest

from offload_runtime.backends.null_backend import NullBackend
from offload_runtime.scheduler.lookahead import LookaheadScheduler
from offload_runtime.scheduler.reverse_scheduler import ReverseLookaheadScheduler
from offload_runtime.storage.in_memory import InMemoryStorage
from offload_runtime.training import TrainingRuntime
from offload_runtime.types import DeviceBuffer, LayerSpec, LoRASpec


class PassthroughTrainingExecutor:
    """Test executor that records forward/backward calls."""

    def __init__(self) -> None:
        self.forward_calls: list[int] = []
        self.backward_calls: list[int] = []

    def forward_layer(
        self, layer: LayerSpec, lora: LoRASpec | None, activations: Any,
        device_weights: DeviceBuffer, lora_a: DeviceBuffer | None,
        lora_b: DeviceBuffer | None, backend: Any, stream: Any,
    ) -> tuple[Any, Any]:
        self.forward_calls.append(layer.layer_id)
        return activations, {"layer_id": layer.layer_id}

    def backward_layer(
        self, layer: LayerSpec, lora: LoRASpec | None, grad_output: Any,
        saved_state: Any, device_weights: DeviceBuffer,
        lora_a: DeviceBuffer | None, lora_b: DeviceBuffer | None,
        backend: Any, stream: Any,
    ) -> tuple[Any, Any, Any]:
        self.backward_calls.append(layer.layer_id)
        return grad_output, None, None


class SumTrainingExecutor:
    """Test executor that accumulates layer_id in forward and decrements in backward."""

    def forward_layer(
        self, layer: LayerSpec, lora: LoRASpec | None, activations: Any,
        device_weights: DeviceBuffer, lora_a: DeviceBuffer | None,
        lora_b: DeviceBuffer | None, backend: Any, stream: Any,
    ) -> tuple[Any, Any]:
        return activations + layer.layer_id, {"layer_id": layer.layer_id}

    def backward_layer(
        self, layer: LayerSpec, lora: LoRASpec | None, grad_output: Any,
        saved_state: Any, device_weights: DeviceBuffer,
        lora_a: DeviceBuffer | None, lora_b: DeviceBuffer | None,
        backend: Any, stream: Any,
    ) -> tuple[Any, Any, Any]:
        return grad_output - layer.layer_id, None, None


def _make_layers(count: int, nbytes: int = 16) -> list[LayerSpec]:
    return [LayerSpec(layer_id=i, name=f"L{i}", nbytes=nbytes) for i in range(count)]


def _make_storage(count: int, nbytes: int = 16) -> InMemoryStorage:
    return InMemoryStorage({i: b"\x00" * nbytes for i in range(count)})


def _make_lora_specs(layer_ids: list[int], rank: int = 4) -> dict[int, LoRASpec]:
    return {
        lid: LoRASpec(layer_id=lid, rank=rank, lora_a_nbytes=rank * 4, lora_b_nbytes=rank * 4)
        for lid in layer_ids
    }


class TestTrainingRuntime:
    def test_forward_backward_order(self) -> None:
        executor = PassthroughTrainingExecutor()
        with TrainingRuntime(
            layers=_make_layers(3),
            lora_specs=_make_lora_specs([0, 1, 2]),
            backend=NullBackend(),
            storage=_make_storage(3),
            forward_scheduler=LookaheadScheduler(lookahead=1),
            backward_scheduler=ReverseLookaheadScheduler(lookahead=1),
            executor=executor,
        ) as runtime:
            runtime.run_training_step([0, 1, 2], inputs=0)

        assert executor.forward_calls == [0, 1, 2]
        assert executor.backward_calls == [2, 1, 0]

    def test_forward_backward_values(self) -> None:
        with TrainingRuntime(
            layers=_make_layers(3),
            lora_specs=_make_lora_specs([0, 1, 2]),
            backend=NullBackend(),
            storage=_make_storage(3),
            forward_scheduler=LookaheadScheduler(lookahead=1),
            backward_scheduler=ReverseLookaheadScheduler(lookahead=1),
            executor=SumTrainingExecutor(),
        ) as runtime:
            grad, metrics = runtime.run_training_step([0, 1, 2], inputs=0)

        # Forward: 0+0+1+2 = 3, Backward: 3-2-1-0 = 0
        assert grad == 0
        assert metrics.layer_count == 3

    def test_lora_buffers_allocated_on_device(self) -> None:
        backend = NullBackend()
        lora_specs = _make_lora_specs([0, 1])
        with TrainingRuntime(
            layers=_make_layers(2),
            lora_specs=lora_specs,
            backend=backend,
            storage=_make_storage(2),
            forward_scheduler=LookaheadScheduler(lookahead=1),
            backward_scheduler=ReverseLookaheadScheduler(lookahead=1),
            executor=PassthroughTrainingExecutor(),
        ) as runtime:
            # LoRA buffers should be allocated (2 layers * 2 buffers each = 4)
            assert len(runtime._lora_buffers) == 2
            for lid in [0, 1]:
                a, b = runtime._lora_buffers[lid]
                assert a.nbytes == lora_specs[lid].lora_a_nbytes
                assert b.nbytes == lora_specs[lid].lora_b_nbytes

    def test_context_manager_cleanup(self) -> None:
        backend = NullBackend()
        runtime = TrainingRuntime(
            layers=_make_layers(2),
            lora_specs=_make_lora_specs([0, 1]),
            backend=backend,
            storage=_make_storage(2),
            forward_scheduler=LookaheadScheduler(lookahead=1),
            backward_scheduler=ReverseLookaheadScheduler(lookahead=1),
            executor=PassthroughTrainingExecutor(),
        )
        runtime.run_training_step([0, 1], inputs=0)
        runtime.close()
        # After close, LoRA buffers freed, device buffers freed
        assert len(runtime._lora_buffers) == 0
        assert len(backend._buffers) == 0

    def test_metrics_include_forward_and_backward(self) -> None:
        with TrainingRuntime(
            layers=_make_layers(3, nbytes=32),
            lora_specs=_make_lora_specs([0, 1, 2]),
            backend=NullBackend(),
            storage=_make_storage(3, nbytes=32),
            forward_scheduler=LookaheadScheduler(lookahead=1),
            backward_scheduler=ReverseLookaheadScheduler(lookahead=1),
            executor=PassthroughTrainingExecutor(),
        ) as runtime:
            _, metrics = runtime.run_training_step([0, 1, 2], inputs=0)

        assert metrics.layer_count == 3
        # Forward + backward = 6 transfers of 32 bytes each
        assert metrics.transferred_bytes == 32 * 6
        assert metrics.end_to_end_seconds > 0
        assert len(metrics.layer_metrics) == 3
        for lm in metrics.layer_metrics:
            assert lm.forward_h2d_ms >= 0
            assert lm.backward_h2d_ms >= 0
            assert lm.forward_compute_ms >= 0
            assert lm.backward_compute_ms >= 0

    def test_no_lora_layers(self) -> None:
        """Runtime should work even if no layers have LoRA adapters."""
        executor = PassthroughTrainingExecutor()
        with TrainingRuntime(
            layers=_make_layers(2),
            lora_specs={},
            backend=NullBackend(),
            storage=_make_storage(2),
            forward_scheduler=LookaheadScheduler(lookahead=1),
            backward_scheduler=ReverseLookaheadScheduler(lookahead=1),
            executor=executor,
        ) as runtime:
            runtime.run_training_step([0, 1], inputs=0)

        assert executor.forward_calls == [0, 1]
        assert executor.backward_calls == [1, 0]

    def test_invalid_layer_id_raises(self) -> None:
        with TrainingRuntime(
            layers=_make_layers(2),
            lora_specs={},
            backend=NullBackend(),
            storage=_make_storage(2),
            forward_scheduler=LookaheadScheduler(lookahead=1),
            backward_scheduler=ReverseLookaheadScheduler(lookahead=1),
            executor=PassthroughTrainingExecutor(),
        ) as runtime:
            with pytest.raises(ValueError, match="Unknown layer IDs"):
                runtime.run_training_step([0, 1, 99], inputs=0)

    def test_with_buffer_pool(self) -> None:
        backend = NullBackend()
        with TrainingRuntime(
            layers=_make_layers(3),
            lora_specs=_make_lora_specs([0, 1, 2]),
            backend=backend,
            storage=_make_storage(3),
            forward_scheduler=LookaheadScheduler(lookahead=1),
            backward_scheduler=ReverseLookaheadScheduler(lookahead=1),
            executor=SumTrainingExecutor(),
            use_buffer_pool=True,
        ) as runtime:
            grad, metrics = runtime.run_training_step([0, 1, 2], inputs=0)
            assert grad == 0
            assert metrics.layer_count == 3
        assert len(backend._buffers) == 0

    def test_partial_lora_some_layers_only(self) -> None:
        """Only some layers have LoRA adapters."""
        executor = PassthroughTrainingExecutor()
        with TrainingRuntime(
            layers=_make_layers(3),
            lora_specs=_make_lora_specs([1]),  # only layer 1
            backend=NullBackend(),
            storage=_make_storage(3),
            forward_scheduler=LookaheadScheduler(lookahead=1),
            backward_scheduler=ReverseLookaheadScheduler(lookahead=1),
            executor=executor,
        ) as runtime:
            runtime.run_training_step([0, 1, 2], inputs=0)

        assert executor.forward_calls == [0, 1, 2]
        assert executor.backward_calls == [2, 1, 0]
        assert len(runtime._lora_buffers) == 0  # cleaned up by close
