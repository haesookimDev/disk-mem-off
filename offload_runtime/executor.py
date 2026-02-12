from __future__ import annotations

from typing import Any

from offload_runtime.backends.base import DeviceBackend
from offload_runtime.types import DeviceBuffer, HostBuffer, LayerSpec


class PassthroughExecutor:
    """Reference executor that validates the data flow without real computation.

    For each layer:
    - Verifies device_weights.nbytes matches layer.nbytes.
    - Optionally reads back device memory via NullBackend to verify H2D copy integrity.
    - Passes activations through unmodified.
    """

    def __init__(self, *, verify_copy: bool = False) -> None:
        self._verify_copy = verify_copy
        self.validated_layers: list[int] = []

    def run_layer(
        self,
        layer: LayerSpec,
        activations: Any,
        device_weights: DeviceBuffer,
        backend: DeviceBackend,
        stream: Any,
    ) -> Any:
        if device_weights.nbytes != layer.nbytes:
            raise RuntimeError(
                f"Layer {layer.layer_id}: device buffer size {device_weights.nbytes} "
                f"!= expected {layer.nbytes}"
            )

        if self._verify_copy and device_weights.backend == "null":
            host_readback = HostBuffer(
                view=memoryview(bytearray(device_weights.nbytes)), pinned=False
            )
            backend.copy_d2h_async(host_readback, device_weights, stream)
            raw = bytes(host_readback.view)
            if raw == b"\x00" * device_weights.nbytes:
                raise RuntimeError(
                    f"Layer {layer.layer_id}: device buffer is all zeros after H2D copy"
                )

        self.validated_layers.append(layer.layer_id)
        return activations
