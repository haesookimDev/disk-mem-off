from __future__ import annotations

from offload_runtime.backends.null_backend import NullBackend
from offload_runtime.buffer_pool import DeviceBufferPool


class TestDeviceBufferPool:
    def setup_method(self) -> None:
        self.backend = NullBackend()
        self.pool = DeviceBufferPool(self.backend)

    def test_acquire_allocates_new(self) -> None:
        buf = self.pool.acquire(64)
        assert buf.nbytes == 64
        assert buf.backend == "null"

    def test_release_and_reuse(self) -> None:
        buf1 = self.pool.acquire(64)
        handle1 = buf1.handle
        self.pool.release(buf1)

        buf2 = self.pool.acquire(64)
        assert buf2.handle == handle1  # reused same buffer

    def test_different_sizes_not_reused(self) -> None:
        buf1 = self.pool.acquire(32)
        self.pool.release(buf1)

        buf2 = self.pool.acquire(64)
        assert buf2.handle != buf1.handle  # different size, new allocation

    def test_drain_frees_all(self) -> None:
        buf1 = self.pool.acquire(32)
        buf2 = self.pool.acquire(32)
        self.pool.release(buf1)
        self.pool.release(buf2)

        assert len(self.backend._buffers) == 2
        self.pool.drain()
        assert len(self.backend._buffers) == 0

    def test_multiple_acquire_release_cycles(self) -> None:
        handles = set()
        for _ in range(5):
            buf = self.pool.acquire(16)
            handles.add(buf.handle)
            self.pool.release(buf)

        # All cycles should reuse the same buffer
        assert len(handles) == 1
