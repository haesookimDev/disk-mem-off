"""NumPy-based executor for GPT-2 architecture."""
from __future__ import annotations

import math
from typing import Any

from offload_runtime.backends.base import DeviceBackend
from offload_runtime.types import DeviceBuffer, LayerSpec

from ._common import (
    _readback_device, _unpack_tensors, gelu, layer_norm, linear, np, softmax,
)

LAYER_TENSORS = [
    "ln_1.weight", "ln_1.bias",
    "attn.c_attn.weight", "attn.c_attn.bias",
    "attn.c_proj.weight", "attn.c_proj.bias",
    "ln_2.weight", "ln_2.bias",
    "mlp.c_fc.weight", "mlp.c_fc.bias",
    "mlp.c_proj.weight", "mlp.c_proj.bias",
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
