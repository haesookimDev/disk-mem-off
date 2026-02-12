from __future__ import annotations

from typing import Any

from .base import DeviceBackend
from offload_runtime.types import DeviceBuffer, HostBuffer


class NullBackend(DeviceBackend):
    """CPU-only backend for dry runs and scheduler/storage testing."""

    name = "null"

    def __init__(self) -> None:
        self._buffers: dict[int, bytearray] = {}

    def create_stream(self, purpose: str) -> object:
        return object()

    def destroy_stream(self, stream: Any) -> None:
        _ = stream

    def alloc_device(self, nbytes: int) -> DeviceBuffer:
        buf = bytearray(nbytes)
        key = id(buf)
        self._buffers[key] = buf
        return DeviceBuffer(handle=key, nbytes=nbytes, backend=self.name)

    def free_device(self, buf: DeviceBuffer) -> None:
        self._buffers.pop(int(buf.handle), None)

    def copy_h2d_async(self, dst: DeviceBuffer, src: HostBuffer, stream: Any) -> None:
        _ = stream
        target = self._buffers[int(dst.handle)]
        src_bytes = src.view.tobytes()
        target[: len(src_bytes)] = src_bytes

    def copy_d2h_async(self, dst: HostBuffer, src: DeviceBuffer, stream: Any) -> None:
        _ = stream
        source = self._buffers[int(src.handle)]
        dst.view[: len(source)] = source

    def record_event(self, stream: Any) -> Any:
        _ = stream
        return object()

    def destroy_event(self, event: Any) -> None:
        _ = event

    def wait_event(self, stream: Any, event: Any) -> None:
        _ = (stream, event)

    def synchronize_stream(self, stream: Any) -> None:
        _ = stream

