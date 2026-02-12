from __future__ import annotations

import pytest

from offload_runtime.backends.null_backend import NullBackend
from offload_runtime.types import DeviceBuffer, HostBuffer


class TestNullBackend:
    def setup_method(self) -> None:
        self.backend = NullBackend()

    def test_create_and_destroy_stream(self) -> None:
        stream = self.backend.create_stream("test")
        assert stream is not None
        self.backend.destroy_stream(stream)

    def test_alloc_and_free_device(self) -> None:
        buf = self.backend.alloc_device(64)
        assert isinstance(buf, DeviceBuffer)
        assert buf.nbytes == 64
        assert buf.backend == "null"
        assert buf.handle in self.backend._buffers
        self.backend.free_device(buf)
        assert buf.handle not in self.backend._buffers

    def test_free_device_idempotent(self) -> None:
        buf = self.backend.alloc_device(32)
        self.backend.free_device(buf)
        self.backend.free_device(buf)  # should not raise

    def test_copy_h2d_async(self) -> None:
        data = bytearray(b"hello world!1234")
        src = HostBuffer(view=memoryview(data), pinned=False)
        dst = self.backend.alloc_device(16)
        stream = self.backend.create_stream("xfer")

        self.backend.copy_h2d_async(dst, src, stream)

        raw = self.backend._buffers[int(dst.handle)]
        assert bytes(raw) == b"hello world!1234"

    def test_copy_d2h_async(self) -> None:
        device_data = bytearray(b"abcdefgh")
        key = id(device_data)
        self.backend._buffers[key] = device_data
        src = DeviceBuffer(handle=key, nbytes=8, backend="null")

        host_buf = bytearray(8)
        dst = HostBuffer(view=memoryview(host_buf), pinned=False)
        stream = self.backend.create_stream("xfer")

        self.backend.copy_d2h_async(dst, src, stream)
        assert bytes(host_buf) == b"abcdefgh"

    def test_record_and_wait_event(self) -> None:
        stream = self.backend.create_stream("s")
        event = self.backend.record_event(stream)
        # NullBackend event is a sentinel; wait_event should not raise
        self.backend.wait_event(stream, event)

    def test_synchronize_stream(self) -> None:
        stream = self.backend.create_stream("s")
        self.backend.synchronize_stream(stream)  # should not raise

    def test_h2d_partial_copy_smaller_src(self) -> None:
        data = bytearray(b"abcd")
        src = HostBuffer(view=memoryview(data), pinned=False)
        dst = self.backend.alloc_device(8)
        stream = self.backend.create_stream("xfer")

        self.backend.copy_h2d_async(dst, src, stream)

        raw = self.backend._buffers[int(dst.handle)]
        assert raw[:4] == bytearray(b"abcd")
