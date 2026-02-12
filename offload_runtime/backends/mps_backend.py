from __future__ import annotations

import ctypes
from typing import Any

from .base import DeviceBackend
from offload_runtime.types import DeviceBuffer, HostBuffer

try:
    import Metal  # type: ignore  # pyobjc-framework-Metal
    _metal_device = Metal.MTLCreateSystemDefaultDevice()
except Exception:  # pragma: no cover - optional dependency
    Metal = None
    _metal_device = None


class MPSBackend(DeviceBackend):
    """Apple Metal Performance Shaders backend for Apple Silicon.

    Key differences from CUDA/ROCm:
    - Unified Memory Architecture: CPU and GPU share physical memory
    - alloc_device creates MTLBuffer with shared storage mode (zero-copy)
    - copy_h2d_async is a memcpy (no DMA transfer), but maintains API consistency
    - Streams map to MTLCommandQueue instances
    - Events map to MTLEvent with counter-based signal/wait
    """

    name = "mps"

    def __init__(self) -> None:
        if Metal is None or _metal_device is None:
            raise RuntimeError(
                "pyobjc-framework-Metal is not installed or no Metal device found; "
                "cannot initialize MPSBackend"
            )
        self._device = _metal_device
        self._event_counter: int = 0
        self._buffers: dict[int, Any] = {}  # handle -> MTLBuffer

    @property
    def supports_pinned_host(self) -> bool:
        return True  # UMA: all host memory is effectively pinned

    @property
    def supports_peer_to_peer(self) -> bool:
        return False  # Single GPU only on Apple Silicon

    @property
    def supports_graph_capture(self) -> bool:
        return False  # Metal does not have CUDA-style graph capture

    def alloc_pinned_host(self, nbytes: int) -> HostBuffer:
        # On UMA, "pinned" is just regular memory
        buf = bytearray(nbytes)
        return HostBuffer(view=memoryview(buf), pinned=True)

    def free_pinned_host(self, buf: HostBuffer) -> None:
        pass  # No special deallocation needed for UMA

    def create_stream(self, purpose: str) -> Any:
        """Create a Metal command queue (analogous to a CUDA stream)."""
        _ = purpose
        queue = self._device.newCommandQueue()
        if queue is None:
            raise RuntimeError("Failed to create Metal command queue")
        return queue

    def destroy_stream(self, stream: Any) -> None:
        # MTLCommandQueue is reference-counted by ObjC runtime
        pass

    def alloc_device(self, nbytes: int) -> DeviceBuffer:
        """Allocate a shared-mode Metal buffer.

        MTLResourceStorageModeShared (= 0) means CPU and GPU can both
        access the buffer. On UMA this is zero-copy.
        """
        mtl_buf = self._device.newBufferWithLength_options_(nbytes, 0)
        if mtl_buf is None:
            raise RuntimeError(f"Failed to allocate {nbytes} bytes on Metal device")
        handle = id(mtl_buf)
        self._buffers[handle] = mtl_buf
        return DeviceBuffer(handle=handle, nbytes=nbytes, backend=self.name)

    def free_device(self, buf: DeviceBuffer) -> None:
        self._buffers.pop(int(buf.handle), None)

    def copy_h2d_async(self, dst: DeviceBuffer, src: HostBuffer, stream: Any) -> None:
        """Copy host data into the shared MTLBuffer."""
        if src.nbytes < dst.nbytes:
            raise ValueError(
                f"Host buffer ({src.nbytes}B) smaller than device buffer ({dst.nbytes}B)"
            )
        mtl_buf = self._buffers[int(dst.handle)]
        contents_ptr = mtl_buf.contents()
        src_bytes = bytes(src.view[: dst.nbytes])
        ctypes.memmove(contents_ptr, src_bytes, dst.nbytes)
        mtl_buf.didModifyRange_(Metal.NSRange(0, dst.nbytes))

    def copy_d2h_async(self, dst: HostBuffer, src: DeviceBuffer, stream: Any) -> None:
        if dst.nbytes < src.nbytes:
            raise ValueError(
                f"Host buffer ({dst.nbytes}B) smaller than device buffer ({src.nbytes}B)"
            )
        mtl_buf = self._buffers[int(src.handle)]
        contents_ptr = mtl_buf.contents()
        raw = ctypes.string_at(contents_ptr, src.nbytes)
        dst.view[: src.nbytes] = raw

    def record_event(self, stream: Any) -> Any:
        """Record an event using MTLEvent signal counter.

        Returns (MTLEvent, counter_value) tuple.
        """
        event = self._device.newEvent()
        if event is None:
            raise RuntimeError("Failed to create Metal event")
        self._event_counter += 1
        counter = self._event_counter
        cmd_buf = stream.commandBuffer()
        cmd_buf.encodeSignalEvent_value_(event, counter)
        cmd_buf.commit()
        return (event, counter)

    def destroy_event(self, event: Any) -> None:
        # MTLEvent is reference-counted; no explicit destroy needed
        pass

    def wait_event(self, stream: Any, event: Any) -> None:
        """Make a command queue wait for an MTLEvent counter."""
        mtl_event, counter = event
        cmd_buf = stream.commandBuffer()
        cmd_buf.encodeWaitForEvent_value_(mtl_event, counter)
        cmd_buf.commit()

    def synchronize_stream(self, stream: Any) -> None:
        """Wait for all commands in the queue to complete."""
        cmd_buf = stream.commandBuffer()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
