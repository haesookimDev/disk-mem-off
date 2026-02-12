from __future__ import annotations

from typing import Any

import pytest

from offload_runtime.backends.null_backend import NullBackend
from offload_runtime.executor import PassthroughExecutor
from offload_runtime.runtime import OffloadRuntime
from offload_runtime.scheduler.lookahead import LookaheadScheduler
from offload_runtime.storage.in_memory import InMemoryStorage
from offload_runtime.types import DeviceBuffer, LayerSpec


class TestPassthroughExecutor:
    def test_validates_sizes(self) -> None:
        backend = NullBackend()
        executor = PassthroughExecutor()
        layer = LayerSpec(layer_id=0, name="L0", nbytes=16)
        device_buf = backend.alloc_device(16)
        stream = backend.create_stream("test")

        executor.run_layer(layer, "act", device_buf, backend, stream)
        assert executor.validated_layers == [0]

    def test_rejects_size_mismatch(self) -> None:
        backend = NullBackend()
        executor = PassthroughExecutor()
        layer = LayerSpec(layer_id=0, name="L0", nbytes=16)
        device_buf = backend.alloc_device(32)  # wrong size
        stream = backend.create_stream("test")

        with pytest.raises(RuntimeError, match="device buffer size"):
            executor.run_layer(layer, "act", device_buf, backend, stream)

    def test_records_validated_layers(self) -> None:
        backend = NullBackend()
        executor = PassthroughExecutor()
        stream = backend.create_stream("test")

        for i in range(3):
            layer = LayerSpec(layer_id=i, name=f"L{i}", nbytes=8)
            device_buf = backend.alloc_device(8)
            executor.run_layer(layer, None, device_buf, backend, stream)

        assert executor.validated_layers == [0, 1, 2]

    def test_preserves_activations(self) -> None:
        backend = NullBackend()
        executor = PassthroughExecutor()
        layer = LayerSpec(layer_id=0, name="L0", nbytes=8)
        device_buf = backend.alloc_device(8)
        stream = backend.create_stream("test")

        result = executor.run_layer(layer, {"key": "value"}, device_buf, backend, stream)
        assert result == {"key": "value"}

    def test_verify_copy_passes_with_nonzero_data(self) -> None:
        backend = NullBackend()
        executor = PassthroughExecutor(verify_copy=True)
        layer = LayerSpec(layer_id=0, name="L0", nbytes=16)

        # Simulate H2D: put non-zero data in the "device" buffer
        device_buf = backend.alloc_device(16)
        src_data = bytearray(b"abcdefghijklmnop")
        from offload_runtime.types import HostBuffer
        src = HostBuffer(view=memoryview(src_data), pinned=False)
        stream = backend.create_stream("test")
        backend.copy_h2d_async(device_buf, src, stream)

        executor.run_layer(layer, None, device_buf, backend, stream)
        assert executor.validated_layers == [0]

    def test_verify_copy_detects_zero_data(self) -> None:
        backend = NullBackend()
        executor = PassthroughExecutor(verify_copy=True)
        layer = LayerSpec(layer_id=0, name="L0", nbytes=16)

        # Device buffer is zero-initialized by NullBackend
        device_buf = backend.alloc_device(16)
        stream = backend.create_stream("test")

        with pytest.raises(RuntimeError, match="all zeros"):
            executor.run_layer(layer, None, device_buf, backend, stream)

    def test_integration_with_runtime(self) -> None:
        layers = [LayerSpec(layer_id=i, name=f"L{i}", nbytes=16) for i in range(3)]
        storage = InMemoryStorage({i: b"x" * 16 for i in range(3)})
        executor = PassthroughExecutor(verify_copy=True)

        with OffloadRuntime(
            layers=layers,
            backend=NullBackend(),
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=executor,
        ) as runtime:
            out, metrics = runtime.run_inference([0, 1, 2], inputs="start")

        assert out == "start"
        assert executor.validated_layers == [0, 1, 2]
        assert metrics.layer_count == 3
