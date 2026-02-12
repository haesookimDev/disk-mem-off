from __future__ import annotations

import ctypes
from typing import Any

from .base import DeviceBackend
from offload_runtime.types import DeviceBuffer, HostBuffer

try:
    from cuda import cudart  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cudart = None


def _check(result: tuple[Any, ...]) -> Any:
    status, *rest = result
    if int(status) != 0:
        raise RuntimeError(f"CUDA runtime call failed with status={int(status)}")
    if not rest:
        return None
    if len(rest) == 1:
        return rest[0]
    return tuple(rest)


def _ptr_from_writable_view(view: memoryview) -> int:
    if view.readonly:
        raise ValueError(
            "HostBuffer.view must be writable for async cudaMemcpyAsync. "
            "Use pinned writable host buffers in production."
        )
    return ctypes.addressof(ctypes.c_char.from_buffer(view))


class CUDABackend(DeviceBackend):
    """Low-level CUDA adapter using cuda-python runtime bindings."""

    name = "cuda"

    def __init__(self, device_id: int = 0) -> None:
        if cudart is None:
            raise RuntimeError("cuda-python is not installed; cannot initialize CUDABackend")
        _check(cudart.cudaSetDevice(device_id))

    def create_stream(self, purpose: str) -> Any:
        _ = purpose
        # cudaStreamNonBlocking = 1
        stream = _check(cudart.cudaStreamCreateWithFlags(1))
        return stream

    def destroy_stream(self, stream: Any) -> None:
        _check(cudart.cudaStreamDestroy(stream))

    def alloc_device(self, nbytes: int) -> DeviceBuffer:
        dev_ptr = _check(cudart.cudaMalloc(nbytes))
        return DeviceBuffer(handle=dev_ptr, nbytes=nbytes, backend=self.name)

    def free_device(self, buf: DeviceBuffer) -> None:
        _check(cudart.cudaFree(buf.handle))

    def copy_h2d_async(self, dst: DeviceBuffer, src: HostBuffer, stream: Any) -> None:
        src_ptr = _ptr_from_writable_view(src.view)
        _check(
            cudart.cudaMemcpyAsync(
                dst.handle,
                src_ptr,
                min(dst.nbytes, src.nbytes),
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            )
        )

    def copy_d2h_async(self, dst: HostBuffer, src: DeviceBuffer, stream: Any) -> None:
        dst_ptr = _ptr_from_writable_view(dst.view)
        _check(
            cudart.cudaMemcpyAsync(
                dst_ptr,
                src.handle,
                min(dst.nbytes, src.nbytes),
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            )
        )

    def record_event(self, stream: Any) -> Any:
        # cudaEventDisableTiming = 2
        event = _check(cudart.cudaEventCreateWithFlags(2))
        _check(cudart.cudaEventRecord(event, stream))
        return event

    def destroy_event(self, event: Any) -> None:
        _check(cudart.cudaEventDestroy(event))

    def wait_event(self, stream: Any, event: Any) -> None:
        _check(cudart.cudaStreamWaitEvent(stream, event, 0))

    def synchronize_stream(self, stream: Any) -> None:
        _check(cudart.cudaStreamSynchronize(stream))

