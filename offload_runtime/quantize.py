from __future__ import annotations

import struct
from typing import Any, Protocol

from offload_runtime.types import HostBuffer, LayerSpec

try:
    import numpy as np
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    np = None


class Dequantizer(Protocol):
    """Converts quantized HostBuffer data to full-precision before H2D transfer."""

    def needs_dequantize(self, layer: LayerSpec) -> bool: ...

    def dequantize(self, layer: LayerSpec, buf: HostBuffer) -> HostBuffer: ...

    def decompressed_nbytes(self, layer: LayerSpec) -> int: ...


class Int8Dequantizer:
    """Dequantizes int8 quantized weights to float32.

    Expects layer.metadata:
      - "dtype": "int8"
      - "scale": float (default 1.0)
      - "zero_point": int (default 0)
    """

    def needs_dequantize(self, layer: LayerSpec) -> bool:
        return layer.metadata.get("dtype") == "int8"

    def decompressed_nbytes(self, layer: LayerSpec) -> int:
        return layer.nbytes * 4  # int8 -> float32

    def dequantize(self, layer: LayerSpec, buf: HostBuffer) -> HostBuffer:
        scale = layer.metadata.get("scale", 1.0)
        zero_point = layer.metadata.get("zero_point", 0)
        if np is not None:
            arr = np.frombuffer(buf.view, dtype=np.int8)
            result = (arr.astype(np.float32) - zero_point) * scale
            return HostBuffer(view=memoryview(result.tobytes()), pinned=False)
        raw = buf.view.tobytes()
        values = struct.unpack(f"{len(raw)}b", raw)
        float_values = [(v - zero_point) * scale for v in values]
        out_bytes = struct.pack(f"{len(float_values)}f", *float_values)
        return HostBuffer(view=memoryview(bytearray(out_bytes)), pinned=False)


class Float16Dequantizer:
    """Dequantizes float16 weights to float32.

    Expects layer.metadata:
      - "dtype": "float16"
    """

    def needs_dequantize(self, layer: LayerSpec) -> bool:
        return layer.metadata.get("dtype") == "float16"

    def decompressed_nbytes(self, layer: LayerSpec) -> int:
        return layer.nbytes * 2  # float16 -> float32

    def dequantize(self, layer: LayerSpec, buf: HostBuffer) -> HostBuffer:
        if np is not None:
            arr = np.frombuffer(buf.view, dtype=np.float16)
            result = arr.astype(np.float32)
            return HostBuffer(view=memoryview(result.tobytes()), pinned=False)
        raw = buf.view.tobytes()
        half_count = len(raw) // 2
        half_values = struct.unpack(f"{half_count}e", raw)
        out_bytes = struct.pack(f"{half_count}f", *half_values)
        return HostBuffer(view=memoryview(bytearray(out_bytes)), pinned=False)


class BFloat16Dequantizer:
    """Dequantizes bfloat16 weights to float32.

    Expects layer.metadata:
      - "dtype": "bfloat16"
    """

    def needs_dequantize(self, layer: LayerSpec) -> bool:
        return layer.metadata.get("dtype") == "bfloat16"

    def decompressed_nbytes(self, layer: LayerSpec) -> int:
        return layer.nbytes * 2  # bfloat16 -> float32

    def dequantize(self, layer: LayerSpec, buf: HostBuffer) -> HostBuffer:
        if np is not None:
            bf16 = np.frombuffer(buf.view, dtype=np.uint16)
            f32 = bf16.astype(np.uint32) << 16
            result = f32.view(np.float32)
            return HostBuffer(view=memoryview(result.tobytes()), pinned=False)
        raw = buf.view.tobytes()
        half_count = len(raw) // 2
        values = struct.unpack(f"<{half_count}H", raw)
        float_bytes = struct.pack(f"<{half_count}I", *(v << 16 for v in values))
        return HostBuffer(view=memoryview(bytearray(float_bytes)), pinned=False)


class CompositeDequantizer:
    """Dispatches to the correct dequantizer based on layer metadata."""

    def __init__(self, dequantizers: list[Any] | None = None) -> None:
        self._dequantizers = dequantizers or [
            Int8Dequantizer(), Float16Dequantizer(), BFloat16Dequantizer(),
        ]

    def needs_dequantize(self, layer: LayerSpec) -> bool:
        return any(d.needs_dequantize(layer) for d in self._dequantizers)

    def decompressed_nbytes(self, layer: LayerSpec) -> int:
        for d in self._dequantizers:
            if d.needs_dequantize(layer):
                return d.decompressed_nbytes(layer)
        return layer.nbytes

    def dequantize(self, layer: LayerSpec, buf: HostBuffer) -> HostBuffer:
        for d in self._dequantizers:
            if d.needs_dequantize(layer):
                return d.dequantize(layer, buf)
        return buf
