from __future__ import annotations

import offload_runtime


class TestExports:
    def test_all_names_importable(self) -> None:
        for name in offload_runtime.__all__:
            attr = getattr(offload_runtime, name, None)
            # CUDABackend may be None on machines without cuda-python
            if name == "CUDABackend":
                continue
            assert attr is not None, f"{name} is not importable from offload_runtime"

    def test_all_names_match_attributes(self) -> None:
        for name in offload_runtime.__all__:
            assert hasattr(offload_runtime, name), f"{name} not found in offload_runtime module"

    def test_expected_public_api(self) -> None:
        expected = {
            "BlockScheduler",
            "CUDABackend",
            "DeviceBackend",
            "DeviceBuffer",
            "DeviceBufferPool",
            "HostBuffer",
            "InMemoryStorage",
            "LayerExecutor",
            "LayerMetrics",
            "LayerSpec",
            "LayerStorage",
            "LookaheadScheduler",
            "NullBackend",
            "OffloadRuntime",
            "PassthroughExecutor",
            "PinnedHostBufferPool",
            "PrefetchScheduler",
            "RuntimeMetrics",
            "ShardedMMapStorage",
        }
        assert set(offload_runtime.__all__) == expected
