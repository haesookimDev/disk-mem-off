from __future__ import annotations

import struct

import pytest

from offload_runtime.quantize import (
    CompositeDequantizer,
    Float16Dequantizer,
    Int8Dequantizer,
)
from offload_runtime.types import HostBuffer, LayerSpec


class TestInt8Dequantizer:
    def setup_method(self) -> None:
        self.dequant = Int8Dequantizer()

    def test_needs_dequantize_true(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=4, metadata={"dtype": "int8", "scale": 0.5})
        assert self.dequant.needs_dequantize(layer) is True

    def test_needs_dequantize_false_no_metadata(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=4)
        assert self.dequant.needs_dequantize(layer) is False

    def test_needs_dequantize_false_different_dtype(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=4, metadata={"dtype": "float16"})
        assert self.dequant.needs_dequantize(layer) is False

    def test_decompressed_nbytes(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=100, metadata={"dtype": "int8"})
        assert self.dequant.decompressed_nbytes(layer) == 400

    def test_dequantize_expands_4x(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=4, metadata={"dtype": "int8", "scale": 1.0})
        buf = HostBuffer(view=memoryview(bytearray(b"\x01\x02\x03\x04")), pinned=False)
        result = self.dequant.dequantize(layer, buf)
        assert result.nbytes == 16  # 4 int8 -> 4 float32

    def test_dequantize_values_correct(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=2, metadata={"dtype": "int8", "scale": 0.5, "zero_point": 0})
        buf = HostBuffer(view=memoryview(bytearray(struct.pack("2b", 4, -2))), pinned=False)
        result = self.dequant.dequantize(layer, buf)
        values = struct.unpack("2f", result.view.tobytes())
        assert values[0] == pytest.approx(2.0)
        assert values[1] == pytest.approx(-1.0)

    def test_dequantize_with_zero_point(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=1, metadata={"dtype": "int8", "scale": 1.0, "zero_point": 10})
        buf = HostBuffer(view=memoryview(bytearray(struct.pack("1b", 15))), pinned=False)
        result = self.dequant.dequantize(layer, buf)
        values = struct.unpack("1f", result.view.tobytes())
        assert values[0] == pytest.approx(5.0)  # (15 - 10) * 1.0


class TestFloat16Dequantizer:
    def setup_method(self) -> None:
        self.dequant = Float16Dequantizer()

    def test_needs_dequantize_true(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=4, metadata={"dtype": "float16"})
        assert self.dequant.needs_dequantize(layer) is True

    def test_needs_dequantize_false(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=4, metadata={"dtype": "int8"})
        assert self.dequant.needs_dequantize(layer) is False

    def test_decompressed_nbytes(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=8, metadata={"dtype": "float16"})
        assert self.dequant.decompressed_nbytes(layer) == 16

    def test_dequantize_expands_2x(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=4, metadata={"dtype": "float16"})
        # Pack two float16 values: 1.0 and 2.0
        raw = struct.pack("2e", 1.0, 2.0)
        buf = HostBuffer(view=memoryview(bytearray(raw)), pinned=False)
        result = self.dequant.dequantize(layer, buf)
        assert result.nbytes == 8  # 2 float16 -> 2 float32
        values = struct.unpack("2f", result.view.tobytes())
        assert values[0] == pytest.approx(1.0)
        assert values[1] == pytest.approx(2.0)


class TestCompositeDequantizer:
    def setup_method(self) -> None:
        self.dequant = CompositeDequantizer()

    def test_dispatches_int8(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=4, metadata={"dtype": "int8", "scale": 1.0})
        assert self.dequant.needs_dequantize(layer) is True
        assert self.dequant.decompressed_nbytes(layer) == 16

    def test_dispatches_float16(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=4, metadata={"dtype": "float16"})
        assert self.dequant.needs_dequantize(layer) is True
        assert self.dequant.decompressed_nbytes(layer) == 8

    def test_passthrough_for_non_quantized(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=16)
        assert self.dequant.needs_dequantize(layer) is False
        assert self.dequant.decompressed_nbytes(layer) == 16
        buf = HostBuffer(view=memoryview(bytearray(16)), pinned=False)
        result = self.dequant.dequantize(layer, buf)
        assert result is buf  # same object, no copy

    def test_dequantize_int8_through_composite(self) -> None:
        layer = LayerSpec(layer_id=0, name="L0", nbytes=2, metadata={"dtype": "int8", "scale": 2.0})
        buf = HostBuffer(view=memoryview(bytearray(struct.pack("2b", 3, -1))), pinned=False)
        result = self.dequant.dequantize(layer, buf)
        values = struct.unpack("2f", result.view.tobytes())
        assert values[0] == pytest.approx(6.0)
        assert values[1] == pytest.approx(-2.0)
