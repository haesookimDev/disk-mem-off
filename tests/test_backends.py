from __future__ import annotations

from unittest import mock

import pytest

from offload_runtime.backends import detect_backend
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

    def test_capability_flags(self) -> None:
        assert self.backend.supports_pinned_host is True
        assert self.backend.supports_peer_to_peer is False
        assert self.backend.supports_graph_capture is False

    def test_custom_backend_capability_override(self) -> None:
        class P2PBackend(NullBackend):
            @property
            def supports_peer_to_peer(self) -> bool:
                return True

        b = P2PBackend()
        assert b.supports_peer_to_peer is True
        assert b.supports_graph_capture is False  # inherited default

    def test_h2d_partial_copy_smaller_src(self) -> None:
        data = bytearray(b"abcd")
        src = HostBuffer(view=memoryview(data), pinned=False)
        dst = self.backend.alloc_device(8)
        stream = self.backend.create_stream("xfer")

        self.backend.copy_h2d_async(dst, src, stream)

        raw = self.backend._buffers[int(dst.handle)]
        assert raw[:4] == bytearray(b"abcd")


class TestDetectBackend:
    def test_fallback_to_null(self) -> None:
        """When no GPU backends are available, detect_backend returns NullBackend."""
        with mock.patch("offload_runtime.backends.CUDABackend", None), \
             mock.patch("offload_runtime.backends.ROCmBackend", None), \
             mock.patch("offload_runtime.backends.MPSBackend", None):
            backend = detect_backend()
            assert isinstance(backend, NullBackend)

    def test_cuda_preferred_over_others(self) -> None:
        """CUDA is tried first when available."""
        sentinel = NullBackend()
        sentinel.name = "cuda"
        fake_cuda = mock.MagicMock(return_value=sentinel)
        with mock.patch("offload_runtime.backends.CUDABackend", fake_cuda), \
             mock.patch("offload_runtime.backends.ROCmBackend", None), \
             mock.patch("offload_runtime.backends.MPSBackend", None):
            backend = detect_backend(device_id=0)
            assert backend.name == "cuda"
            fake_cuda.assert_called_once_with(0)

    def test_rocm_when_cuda_unavailable(self) -> None:
        """ROCm is tried when CUDA import is None."""
        sentinel = NullBackend()
        sentinel.name = "rocm"
        fake_rocm = mock.MagicMock(return_value=sentinel)
        with mock.patch("offload_runtime.backends.CUDABackend", None), \
             mock.patch("offload_runtime.backends.ROCmBackend", fake_rocm), \
             mock.patch("offload_runtime.backends.MPSBackend", None):
            backend = detect_backend(device_id=1)
            assert backend.name == "rocm"
            fake_rocm.assert_called_once_with(1)

    def test_mps_when_cuda_and_rocm_unavailable(self) -> None:
        """MPS is tried when CUDA and ROCm are None."""
        sentinel = NullBackend()
        sentinel.name = "mps"
        fake_mps = mock.MagicMock(return_value=sentinel)
        with mock.patch("offload_runtime.backends.CUDABackend", None), \
             mock.patch("offload_runtime.backends.ROCmBackend", None), \
             mock.patch("offload_runtime.backends.MPSBackend", fake_mps):
            backend = detect_backend()
            assert backend.name == "mps"
            fake_mps.assert_called_once_with()

    def test_skips_backend_that_raises(self) -> None:
        """If a backend class exists but init raises, skip to the next."""
        fake_cuda = mock.MagicMock(side_effect=RuntimeError("no GPU"))
        with mock.patch("offload_runtime.backends.CUDABackend", fake_cuda), \
             mock.patch("offload_runtime.backends.ROCmBackend", None), \
             mock.patch("offload_runtime.backends.MPSBackend", None):
            backend = detect_backend()
            assert isinstance(backend, NullBackend)
