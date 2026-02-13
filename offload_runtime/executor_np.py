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


# ---------------------------------------------------------------------------
# GPT-2 Executor
# ---------------------------------------------------------------------------

_GPT2_TENSOR_ORDER = [
    "ln_1.weight", "ln_1.bias",
    "attn.c_attn.weight", "attn.c_attn.bias",
    "attn.c_proj.weight", "attn.c_proj.bias",
    "ln_2.weight", "ln_2.bias",
    "mlp.c_fc.weight", "mlp.c_fc.bias",
    "mlp.c_proj.weight", "mlp.c_proj.bias",
]

_LLAMA_TENSOR_ORDER = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

_GLM4_TENSOR_ORDER = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.q_proj.bias",
    "self_attn.k_proj.weight", "self_attn.k_proj.bias",
    "self_attn.v_proj.weight", "self_attn.v_proj.bias",
    "self_attn.o_proj.weight", "self_attn.o_proj.bias",
    "post_attention_layernorm.weight",
    "mlp.gate_up_proj.weight",
    "mlp.down_proj.weight",
    "post_mlp_layernorm.weight",
]


class GPT2Executor:
    """NumPy-based executor for GPT-2 architecture."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.n_embd: int = config["n_embd"]
        self.n_head: int = config["n_head"]
        self.n_positions: int = config.get("n_positions", 1024)

    # -- LayerExecutor protocol --

    def run_layer(
        self,
        layer: LayerSpec,
        activations: Any,
        device_weights: DeviceBuffer,
        backend: DeviceBackend,
        stream: Any,
    ) -> Any:
        raw = _readback_device(device_weights, backend, stream)
        t = _unpack_tensors(raw, layer.metadata["tensors"])

        x: Any = activations
        seq_len = x.shape[0]
        n_embd = self.n_embd
        n_head = self.n_head
        head_dim = n_embd // n_head

        # --- Attention block ---
        h = layer_norm(x, t["ln_1.weight"], t["ln_1.bias"])
        qkv = linear(h, t["attn.c_attn.weight"], t["attn.c_attn.bias"])
        q, k, v = np.split(qkv, 3, axis=-1)

        q = q.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)

        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_dim)
        mask = np.triu(np.full((seq_len, seq_len), -1e10, dtype=np.float32), k=1)
        scores = scores + mask
        attn_w = softmax(scores, axis=-1)
        attn_out = attn_w @ v

        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, n_embd)
        attn_out = linear(attn_out, t["attn.c_proj.weight"], t["attn.c_proj.bias"])
        x = x + attn_out

        # --- MLP block ---
        h = layer_norm(x, t["ln_2.weight"], t["ln_2.bias"])
        h = gelu(linear(h, t["mlp.c_fc.weight"], t["mlp.c_fc.bias"]))
        h = linear(h, t["mlp.c_proj.weight"], t["mlp.c_proj.bias"])
        x = x + h

        return x

    # -- Non-layer helpers (called outside OffloadRuntime) --

    def embed(
        self, token_ids: list[int], wte: Any, wpe: Any,
    ) -> Any:
        seq_len = len(token_ids)
        return wte[token_ids] + wpe[np.arange(seq_len)]

    def lm_head(
        self, hidden: Any, ln_f_w: Any, ln_f_b: Any, head_w: Any,
    ) -> Any:
        h = layer_norm(hidden, ln_f_w, ln_f_b)
        return h @ head_w.T


# ---------------------------------------------------------------------------
# LLaMA Executor
# ---------------------------------------------------------------------------

class LlamaExecutor:
    """NumPy-based executor for LLaMA architecture."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.hidden_size: int = config["hidden_size"]
        self.num_heads: int = config["num_attention_heads"]
        self.num_kv_heads: int = config.get("num_key_value_heads", self.num_heads)
        self.head_dim: int = self.hidden_size // self.num_heads
        self.intermediate_size: int = config["intermediate_size"]
        self.rms_norm_eps: float = config.get("rms_norm_eps", 1e-6)
        self.rope_theta: float = config.get("rope_theta", 10000.0)

    # -- LayerExecutor protocol --

    def run_layer(
        self,
        layer: LayerSpec,
        activations: Any,
        device_weights: DeviceBuffer,
        backend: DeviceBackend,
        stream: Any,
    ) -> Any:
        raw = _readback_device(device_weights, backend, stream)
        t = _unpack_tensors(raw, layer.metadata["tensors"])

        x: Any = activations
        seq_len = x.shape[0]
        n_heads = self.num_heads
        n_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        # --- Attention block ---
        h = rms_norm(x, t["input_layernorm.weight"], self.rms_norm_eps)

        q = linear_t(h, t["self_attn.q_proj.weight"]).reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
        k = linear_t(h, t["self_attn.k_proj.weight"]).reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)
        v = linear_t(h, t["self_attn.v_proj.weight"]).reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)

        positions = np.arange(seq_len, dtype=np.float32)
        q, k = rope(q, k, positions, head_dim, self.rope_theta)

        n_rep = n_heads // n_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_dim)
        mask = np.triu(np.full((seq_len, seq_len), -1e10, dtype=np.float32), k=1)
        scores = scores + mask
        attn_w = softmax(scores, axis=-1)
        attn_out = attn_w @ v

        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, self.hidden_size)
        attn_out = linear_t(attn_out, t["self_attn.o_proj.weight"])
        x = x + attn_out

        # --- SwiGLU MLP block ---
        h = rms_norm(x, t["post_attention_layernorm.weight"], self.rms_norm_eps)
        gate = linear_t(h, t["mlp.gate_proj.weight"])
        up = linear_t(h, t["mlp.up_proj.weight"])
        h = silu(gate) * up
        h = linear_t(h, t["mlp.down_proj.weight"])
        x = x + h

        return x

    # -- Non-layer helpers --

    def embed(self, token_ids: list[int], embed_weights: Any) -> Any:
        return embed_weights[token_ids]

    def lm_head(self, hidden: Any, norm_w: Any, head_w: Any) -> Any:
        h = rms_norm(hidden, norm_w, self.rms_norm_eps)
        return h @ head_w.T


# ---------------------------------------------------------------------------
# GLM-4 Executor
# ---------------------------------------------------------------------------

class GLM4Executor:
    """NumPy-based executor for GLM-4 architecture.

    Key differences from LLaMA:
    - Fused gate_up_proj instead of separate gate_proj + up_proj
    - Attention projections with bias (when attention_bias=True)
    - Additional post_mlp_layernorm after MLP block
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.hidden_size: int = config["hidden_size"]
        self.num_heads: int = config["num_attention_heads"]
        self.num_kv_heads: int = config.get("num_key_value_heads", self.num_heads)
        self.head_dim: int = config.get("head_dim", self.hidden_size // self.num_heads)
        self.intermediate_size: int = config["intermediate_size"]
        self.rms_norm_eps: float = config.get("rms_norm_eps", 1.5625e-7)
        self.rope_theta: float = config.get("rope_theta", 10000.0)

    # -- LayerExecutor protocol --

    def run_layer(
        self,
        layer: LayerSpec,
        activations: Any,
        device_weights: DeviceBuffer,
        backend: DeviceBackend,
        stream: Any,
    ) -> Any:
        raw = _readback_device(device_weights, backend, stream)
        t = _unpack_tensors(raw, layer.metadata["tensors"])

        x: Any = activations
        seq_len = x.shape[0]
        n_heads = self.num_heads
        n_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        # --- Attention block ---
        h = rms_norm(x, t["input_layernorm.weight"], self.rms_norm_eps)

        q = linear_t(h, t["self_attn.q_proj.weight"])
        if "self_attn.q_proj.bias" in t:
            q = q + t["self_attn.q_proj.bias"]
        k = linear_t(h, t["self_attn.k_proj.weight"])
        if "self_attn.k_proj.bias" in t:
            k = k + t["self_attn.k_proj.bias"]
        v = linear_t(h, t["self_attn.v_proj.weight"])
        if "self_attn.v_proj.bias" in t:
            v = v + t["self_attn.v_proj.bias"]

        q = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)

        positions = np.arange(seq_len, dtype=np.float32)
        q, k = rope(q, k, positions, head_dim, self.rope_theta)

        n_rep = n_heads // n_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_dim)
        mask = np.triu(np.full((seq_len, seq_len), -1e10, dtype=np.float32), k=1)
        scores = scores + mask
        attn_w = softmax(scores, axis=-1)
        attn_out = attn_w @ v

        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)
        attn_out = linear_t(attn_out, t["self_attn.o_proj.weight"])
        if "self_attn.o_proj.bias" in t:
            attn_out = attn_out + t["self_attn.o_proj.bias"]
        x = x + attn_out

        # --- SwiGLU MLP block (fused gate_up_proj) ---
        h = rms_norm(x, t["post_attention_layernorm.weight"], self.rms_norm_eps)
        gate_up = linear_t(h, t["mlp.gate_up_proj.weight"])
        gate, up = np.split(gate_up, 2, axis=-1)
        h = silu(gate) * up
        h = linear_t(h, t["mlp.down_proj.weight"])

        # Post-MLP layernorm (GLM-4 specific)
        if "post_mlp_layernorm.weight" in t:
            h = rms_norm(h, t["post_mlp_layernorm.weight"], self.rms_norm_eps)

        x = x + h

        return x

    # -- Non-layer helpers --

    def embed(self, token_ids: list[int], embed_weights: Any) -> Any:
        return embed_weights[token_ids]

    def lm_head(self, hidden: Any, norm_w: Any, head_w: Any) -> Any:
        h = rms_norm(hidden, norm_w, self.rms_norm_eps)
        return h @ head_w.T
