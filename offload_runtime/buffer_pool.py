from __future__ import annotations

from collections import defaultdict

from offload_runtime.backends.base import DeviceBackend
from offload_runtime.types import DeviceBuffer


class DeviceBufferPool:
    """Reuses device buffers of the same size to avoid frequent malloc/free."""

    def __init__(self, backend: DeviceBackend) -> None:
        self._backend = backend
        self._free: dict[int, list[DeviceBuffer]] = defaultdict(list)

    def acquire(self, nbytes: int) -> DeviceBuffer:
        pool = self._free.get(nbytes)
        if pool:
            return pool.pop()
        return self._backend.alloc_device(nbytes)

    def release(self, buf: DeviceBuffer) -> None:
        self._free[buf.nbytes].append(buf)

    def drain(self) -> None:
        for pool in self._free.values():
            for buf in pool:
                self._backend.free_device(buf)
            pool.clear()
        self._free.clear()
