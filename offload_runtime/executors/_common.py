"""Shared math primitives and low-level helpers for NumPy-based executors."""
from __future__ import annotations

import math
from typing import Any

from offload_runtime.backends.base import DeviceBackend
from offload_runtime.types import DeviceBuffer, HostBuffer, LayerSpec

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _readback_device(
    device_weights: DeviceBuffer,
    backend: DeviceBackend,
    stream: Any,
) -> bytes:
    """Copy bytes back from device memory so we can work with them on the host."""
    readback = HostBuffer(
        view=memoryview(bytearray(device_weights.nbytes)),
        pinned=False,
    )
    backend.copy_d2h_async(readback, device_weights, stream)
    return bytes(readback.view)


def _unpack_tensors(
    raw_bytes: bytes,
    tensor_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    """Unpack contiguous raw bytes into named numpy arrays using metadata."""
    tensors: dict[str, Any] = {}
    for meta in tensor_meta:
        dtype = np.dtype(meta["dtype"])
        offset = meta["offset"]
        nbytes = meta["nbytes"]
        shape = meta["shape"]
        arr = np.frombuffer(
            raw_bytes, dtype=dtype, count=nbytes // dtype.itemsize, offset=offset,
        )
        tensors[meta["name"]] = arr.reshape(shape)
    return tensors


# ---------------------------------------------------------------------------
# Shared math primitives
# ---------------------------------------------------------------------------

def layer_norm(
    x: Any, weight: Any, bias: Any, eps: float = 1e-5,
) -> Any:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def rms_norm(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return weight * (x / rms)


def gelu(x: Any) -> Any:
    """Approximate GELU (tanh variant used by GPT-2)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def silu(x: Any) -> Any:
    return x / (1.0 + np.exp(-x))


def softmax(x: Any, axis: int = -1) -> Any:
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def linear(x: Any, weight: Any, bias: Any | None = None) -> Any:
    """GPT-2 Conv1D convention: x @ weight + bias (weight is [in, out])."""
    out = x @ weight
    if bias is not None:
        out = out + bias
    return out


def linear_t(x: Any, weight: Any) -> Any:
    """Standard nn.Linear convention: x @ weight.T (weight is [out, in])."""
    return x @ weight.T


def rope(
    q: Any, k: Any, positions: Any, head_dim: int, base: float = 10000.0,
) -> tuple[Any, Any]:
    """Apply Rotary Position Embeddings to query and key tensors.

    q, k: [n_heads, seq_len, head_dim]
    positions: [seq_len]
    """
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (np.arange(0, half_dim, dtype=np.float64) / half_dim))
    angles = np.outer(positions.astype(np.float64), freqs).astype(np.float32)
    cos_vals = np.cos(angles)  # [seq_len, half_dim]
    sin_vals = np.sin(angles)

    def _rotate(x: Any) -> Any:
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return np.concatenate(
            [x1 * cos_vals - x2 * sin_vals, x2 * cos_vals + x1 * sin_vals],
            axis=-1,
        )

    return _rotate(q), _rotate(k)


def repeat_kv(x: Any, n_rep: int) -> Any:
    """Repeat KV heads to match Q heads.  x: [n_kv_heads, seq_len, head_dim]."""
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=0)
