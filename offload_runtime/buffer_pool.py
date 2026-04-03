from __future__ import annotations

from collections import defaultdict

from offload_runtime.backends.base import DeviceBackend
from offload_runtime.types import DeviceBuffer


_POOL_ALIGNMENT = 256


def _align_size(nbytes: int) -> int:
    """Round up to nearest multiple of _POOL_ALIGNMENT for better pool reuse."""
    return (nbytes + _POOL_ALIGNMENT - 1) & ~(_POOL_ALIGNMENT - 1)


class DeviceBufferPool:
    """Reuses device buffers of the same size to avoid frequent malloc/free."""

    def __init__(self, backend: DeviceBackend) -> None:
        self._backend = backend
        self._free: dict[int, list[DeviceBuffer]] = defaultdict(list)

    def acquire(self, nbytes: int) -> DeviceBuffer:
        aligned = _align_size(nbytes)
        pool = self._free.get(aligned)
        if pool:
            return pool.pop()
        return self._backend.alloc_device(aligned)

    def release(self, buf: DeviceBuffer) -> None:
        self._free[buf.nbytes].append(buf)

    def drain(self) -> None:
        for pool in self._free.values():
            for buf in pool:
                self._backend.free_device(buf)
            pool.clear()
        self._free.clear()
