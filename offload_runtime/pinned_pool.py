from __future__ import annotations

from collections import defaultdict

from offload_runtime.backends.base import DeviceBackend
from offload_runtime.types import HostBuffer


class PinnedHostBufferPool:
    """Reuses pinned host buffers of the same size to avoid frequent alloc/free."""

    def __init__(self, backend: DeviceBackend) -> None:
        self._backend = backend
        self._free: dict[int, list[HostBuffer]] = defaultdict(list)

    def acquire(self, nbytes: int) -> HostBuffer:
        pool = self._free.get(nbytes)
        if pool:
            return pool.pop()
        return self._backend.alloc_pinned_host(nbytes)

    def release(self, buf: HostBuffer) -> None:
        self._free[buf.nbytes].append(buf)

    def drain(self) -> None:
        for pool in self._free.values():
            for buf in pool:
                self._backend.free_pinned_host(buf)
            pool.clear()
        self._free.clear()
