from __future__ import annotations

from .base import DeviceBackend
from .null_backend import NullBackend

try:
    from .cuda_backend import CUDABackend
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    CUDABackend = None

try:
    from .rocm_backend import ROCmBackend
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    ROCmBackend = None

try:
    from .mps_backend import MPSBackend
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    MPSBackend = None


def detect_backend(device_id: int = 0) -> DeviceBackend:
    """Auto-detect the best available backend: CUDA → ROCm → MPS → Null."""
    for BackendClass in (CUDABackend, ROCmBackend, MPSBackend):
        if BackendClass is None:
            continue
        try:
            return BackendClass() if BackendClass is MPSBackend else BackendClass(device_id)
        except Exception:  # pragma: no cover - hardware not available
            continue
    return NullBackend()


__all__ = [
    "CUDABackend", "DeviceBackend", "MPSBackend", "NullBackend", "ROCmBackend",
    "detect_backend",
]
