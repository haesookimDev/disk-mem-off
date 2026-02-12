from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from offload_runtime.types import DeviceBuffer, HostBuffer


class DeviceBackend(ABC):
    """Vendor-specific device API adapter.

    Core runtime depends only on this contract, so CUDA/ROCm/etc.
    can be swapped without changing scheduling/offload logic.
    """

    name: str

    @abstractmethod
    def create_stream(self, purpose: str) -> Any:
        pass

    @abstractmethod
    def destroy_stream(self, stream: Any) -> None:
        pass

    @abstractmethod
    def alloc_device(self, nbytes: int) -> DeviceBuffer:
        pass

    @abstractmethod
    def free_device(self, buf: DeviceBuffer) -> None:
        pass

    @abstractmethod
    def copy_h2d_async(self, dst: DeviceBuffer, src: HostBuffer, stream: Any) -> None:
        pass

    @abstractmethod
    def copy_d2h_async(self, dst: HostBuffer, src: DeviceBuffer, stream: Any) -> None:
        pass

    @abstractmethod
    def record_event(self, stream: Any) -> Any:
        pass

    @abstractmethod
    def destroy_event(self, event: Any) -> None:
        pass

    @abstractmethod
    def wait_event(self, stream: Any, event: Any) -> None:
        pass

    @abstractmethod
    def synchronize_stream(self, stream: Any) -> None:
        pass

