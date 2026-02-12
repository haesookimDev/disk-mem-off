from __future__ import annotations

import pytest

from offload_runtime.backends import ROCmBackend


class TestROCmBackendStructure:
    def test_conditional_import_does_not_raise(self) -> None:
        # ROCmBackend may be None on non-ROCm machines -- that's expected
        from offload_runtime.backends import ROCmBackend as RB
        assert RB is None or callable(RB)

    def test_module_defines_class(self) -> None:
        from offload_runtime.backends import rocm_backend
        assert hasattr(rocm_backend, "ROCmBackend")


@pytest.mark.skipif(ROCmBackend is None, reason="hip-python not available")
class TestROCmBackend:
    def setup_method(self) -> None:
        self.backend = ROCmBackend(device_id=0)

    def test_name(self) -> None:
        assert self.backend.name == "rocm"

    def test_capability_flags(self) -> None:
        assert self.backend.supports_pinned_host is True
        assert self.backend.supports_peer_to_peer is True
        assert self.backend.supports_graph_capture is True

    def test_create_and_destroy_stream(self) -> None:
        stream = self.backend.create_stream("test")
        assert stream is not None
        self.backend.destroy_stream(stream)

    def test_alloc_and_free_device(self) -> None:
        from offload_runtime.types import DeviceBuffer
        buf = self.backend.alloc_device(64)
        assert isinstance(buf, DeviceBuffer)
        assert buf.nbytes == 64
        assert buf.backend == "rocm"
        self.backend.free_device(buf)

    def test_pinned_host_alloc_free(self) -> None:
        buf = self.backend.alloc_pinned_host(64)
        assert buf.pinned is True
        assert buf.nbytes == 64
        self.backend.free_pinned_host(buf)

    def test_h2d_d2h_roundtrip(self) -> None:
        from offload_runtime.types import HostBuffer
        data = bytearray(b"rocm test data!!")[:16]
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
