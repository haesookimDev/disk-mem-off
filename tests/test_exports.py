from __future__ import annotations

import offload_runtime


class TestExports:
    # Symbols that may be None due to missing optional dependencies
    _optional_exports = {
        "CUDABackend", "ROCmBackend", "MPSBackend",
        "GPT2Executor", "LlamaExecutor", "Qwen3NextExecutor",
        "HuggingFaceLoader", "ModelBundle", "SafetensorsStorage",
    }

    def test_all_names_importable(self) -> None:
        for name in offload_runtime.__all__:
            attr = getattr(offload_runtime, name, None)
            if name in self._optional_exports:
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
            "detect_backend",
            "DeviceBackend",
            "DeviceBuffer",
            "DeviceBufferPool",
            "Float16Dequantizer",
            "GLM4Executor",
            "GLM4MoeExecutor",
            "GPT2Executor",
            "HostBuffer",
            "HuggingFaceLoader",
            "InMemoryStorage",
            "Int8Dequantizer",
            "LayerExecutor",
            "LayerFeedback",
            "LayerMetrics",
            "LayerSizeInfo",
            "LayerSpec",
            "LayerStorage",
            "LlamaExecutor",
            "LoRASpec",
            "LookaheadScheduler",
            "ModelBundle",
            "MPSBackend",
            "NullBackend",
            "OffloadRuntime",
            "PassthroughExecutor",
            "PinnedHostBufferPool",
            "PrefetchScheduler",
            "Qwen3NextExecutor",
            "ResourceAwareScheduler",
            "ResourceContext",
            "ResourceSnapshot",
            "ReverseLookaheadScheduler",
            "ROCmBackend",
            "RuntimeMetrics",
            "SafetensorsStorage",
            "ShardedMMapStorage",
            "TrainingExecutor",
            "TrainingLayerMetrics",
            "TrainingMetrics",
            "TrainingRuntime",
        }
        assert set(offload_runtime.__all__) == expected
