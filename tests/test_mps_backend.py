from __future__ import annotations

import platform

import pytest

from offload_runtime.backends import MPSBackend
from offload_runtime.backends.mps_backend import Metal as _Metal, _metal_device

_has_metal = _Metal is not None and _metal_device is not None


class TestMPSBackendStructure:
    def test_conditional_import_does_not_raise(self) -> None:
        from offload_runtime.backends import MPSBackend as MB
        assert MB is None or callable(MB)

    def test_module_defines_class(self) -> None:
        from offload_runtime.backends import mps_backend
        assert hasattr(mps_backend, "MPSBackend")

    def test_init_raises_without_metal(self) -> None:
        if _has_metal:
            pytest.skip("Metal is available")
        with pytest.raises(RuntimeError, match="pyobjc-framework-Metal is not installed"):
            MPSBackend()


@pytest.mark.skipif(not _has_metal, reason="Metal/MPS not available")
class TestMPSBackend:
    def setup_method(self) -> None:
        self.backend = MPSBackend()

    def test_name(self) -> None:
        assert self.backend.name == "mps"

    def test_capability_flags(self) -> None:
        assert self.backend.supports_pinned_host is True
        assert self.backend.supports_peer_to_peer is False
        assert self.backend.supports_graph_capture is False

    def test_create_destroy_stream(self) -> None:
        stream = self.backend.create_stream("test")
        assert stream is not None
        self.backend.destroy_stream(stream)

    def test_alloc_free_device(self) -> None:
        from offload_runtime.types import DeviceBuffer
        buf = self.backend.alloc_device(256)
        assert isinstance(buf, DeviceBuffer)
        assert buf.nbytes == 256
        assert buf.backend == "mps"
        self.backend.free_device(buf)

    def test_h2d_d2h_roundtrip(self) -> None:
        from offload_runtime.types import HostBuffer
        data = bytearray(b"hello mps metal!")
        src = HostBuffer(view=memoryview(bytearray(data)), pinned=False)
        device_buf = self.backend.alloc_device(16)
        stream = self.backend.create_stream("xfer")

        self.backend.copy_h2d_async(device_buf, src, stream)

        dst = HostBuffer(view=memoryview(bytearray(16)), pinned=False)
        self.backend.copy_d2h_async(dst, device_buf, stream)
        self.backend.synchronize_stream(stream)

        assert bytes(dst.view) == bytes(data)

        self.backend.free_device(device_buf)
        self.backend.destroy_stream(stream)

    def test_event_record_and_wait(self) -> None:
        s1 = self.backend.create_stream("s1")
        s2 = self.backend.create_stream("s2")
        event = self.backend.record_event(s1)
        self.backend.wait_event(s2, event)
        self.backend.destroy_event(event)
        self.backend.destroy_stream(s1)
        self.backend.destroy_stream(s2)

    def test_pinned_host_alloc_free(self) -> None:
        buf = self.backend.alloc_pinned_host(64)
        assert buf.pinned is True
        assert buf.nbytes == 64
        self.backend.free_pinned_host(buf)

    def test_size_validation_h2d(self) -> None:
        from offload_runtime.types import HostBuffer
        small_src = HostBuffer(view=memoryview(bytearray(4)), pinned=False)
        big_dst = self.backend.alloc_device(16)
        stream = self.backend.create_stream("xfer")
        with pytest.raises(ValueError, match="smaller than device buffer"):
            self.backend.copy_h2d_async(big_dst, small_src, stream)
        self.backend.free_device(big_dst)
        self.backend.destroy_stream(stream)
