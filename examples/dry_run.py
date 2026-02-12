from __future__ import annotations

from offload_runtime.backends.null_backend import NullBackend
from offload_runtime.runtime import OffloadRuntime
from offload_runtime.scheduler.lookahead import LookaheadScheduler
from offload_runtime.storage.in_memory import InMemoryStorage
from offload_runtime.types import DeviceBuffer, LayerSpec


class DummyExecutor:
    def run_layer(self, layer: LayerSpec, activations: int, device_weights: DeviceBuffer, backend, stream) -> int:
        _ = (device_weights, backend, stream)
        return activations + layer.layer_id


def main() -> None:
    layers = [
        LayerSpec(layer_id=0, name="L0", nbytes=16),
        LayerSpec(layer_id=1, name="L1", nbytes=16),
        LayerSpec(layer_id=2, name="L2", nbytes=16),
    ]
    storage = InMemoryStorage(
        layer_bytes={
            0: b"a" * 16,
            1: b"b" * 16,
            2: b"c" * 16,
        }
    )
    runtime = OffloadRuntime(
        layers=layers,
        backend=NullBackend(),
        storage=storage,
        scheduler=LookaheadScheduler(lookahead=2),
        executor=DummyExecutor(),
    )
    out, metrics = runtime.run_inference([0, 1, 2], inputs=0)
    print("output:", out)
    print("metrics:", metrics)


if __name__ == "__main__":
    main()

