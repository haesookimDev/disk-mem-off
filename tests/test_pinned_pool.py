from __future__ import annotations

from offload_runtime.backends.null_backend import NullBackend
from offload_runtime.pinned_pool import PinnedHostBufferPool


class TestPinnedHostBufferPool:
    def setup_method(self) -> None:
        self.backend = NullBackend()
        self.pool = PinnedHostBufferPool(self.backend)

    def test_acquire_allocates_pinned(self) -> None:
        buf = self.pool.acquire(64)
        assert buf.pinned is True
        assert buf.nbytes == 64

    def test_release_and_reuse(self) -> None:
        buf1 = self.pool.acquire(64)
        obj1 = buf1.view.obj
        self.pool.release(buf1)

        buf2 = self.pool.acquire(64)
        assert buf2.view.obj is obj1  # reused same underlying buffer

    def test_different_sizes_not_reused(self) -> None:
        buf1 = self.pool.acquire(32)
        self.pool.release(buf1)

        buf2 = self.pool.acquire(64)
        assert buf2.nbytes == 64
        assert buf2.view.obj is not buf1.view.obj

    def test_drain(self) -> None:
        buf1 = self.pool.acquire(32)
        buf2 = self.pool.acquire(32)
        self.pool.release(buf1)
        self.pool.release(buf2)
        self.pool.drain()
        # After drain, pool should be empty
        assert len(self.pool._free) == 0

    def test_multiple_cycles(self) -> None:
        objs = set()
        for _ in range(5):
            buf = self.pool.acquire(16)
            objs.add(id(buf.view.obj))
            self.pool.release(buf)
        assert len(objs) == 1  # always reused the same buffer
