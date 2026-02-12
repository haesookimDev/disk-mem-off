from __future__ import annotations

import offload_runtime


class TestExports:
    # Backends that may be None due to missing optional dependencies
    _optional_backends = {"CUDABackend", "ROCmBackend", "MPSBackend"}

    def test_all_names_importable(self) -> None:
        for name in offload_runtime.__all__:
            attr = getattr(offload_runtime, name, None)
            if name in self._optional_backends:
                continue
            assert attr is not None, f"{name} is not importable from offload_runtime"

    def test_all_names_match_attributes(self) -> None:
        for name in offload_runtime.__all__:
            assert hasattr(offload_runtime, name), f"{name} not found in offload_runtime module"

    def test_expected_public_api(self) -> None:
        expected = {
            "BlockScheduler",
            "CompositeDequantizer",
            "CostAwareScheduler",
            "CUDABackend",
            "Dequantizer",
            "DeviceBackend",
            "DeviceBuffer",
            "DeviceBufferPool",
            "Float16Dequantizer",
            "HostBuffer",
            "InMemoryStorage",
            "Int8Dequantizer",
            "LayerExecutor",
            "LayerMetrics",
            "LayerSpec",
            "LayerStorage",
            "LoRASpec",
            "LookaheadScheduler",
            "MPSBackend",
            "NullBackend",
            "OffloadRuntime",
            "PassthroughExecutor",
            "PinnedHostBufferPool",
            "PrefetchScheduler",
            "ReverseLookaheadScheduler",
            "ROCmBackend",
            "RuntimeMetrics",
            "ShardedMMapStorage",
            "TrainingExecutor",
            "TrainingLayerMetrics",
            "TrainingMetrics",
            "TrainingRuntime",
        }
        assert set(offload_runtime.__all__) == expected
