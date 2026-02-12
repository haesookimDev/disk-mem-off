from __future__ import annotations

import struct
from typing import Any, Protocol

from offload_runtime.types import HostBuffer, LayerSpec


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
        raw = buf.view.tobytes()
        half_count = len(raw) // 2
        half_values = struct.unpack(f"{half_count}e", raw)
        out_bytes = struct.pack(f"{half_count}f", *half_values)
        return HostBuffer(view=memoryview(bytearray(out_bytes)), pinned=False)


class CompositeDequantizer:
    """Dispatches to the correct dequantizer based on layer metadata."""

    def __init__(self, dequantizers: list[Any] | None = None) -> None:
        self._dequantizers = dequantizers or [Int8Dequantizer(), Float16Dequantizer()]

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
