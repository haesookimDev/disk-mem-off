from .base import DeviceBackend
from .null_backend import NullBackend

try:
    from .cuda_backend import CUDABackend
except Exception:  # pragma: no cover - optional dependency
    CUDABackend = None

try:
    from .rocm_backend import ROCmBackend
except Exception:  # pragma: no cover - optional dependency
    ROCmBackend = None

try:
    from .mps_backend import MPSBackend
except Exception:  # pragma: no cover - optional dependency
    MPSBackend = None

__all__ = ["CUDABackend", "DeviceBackend", "MPSBackend", "NullBackend", "ROCmBackend"]
