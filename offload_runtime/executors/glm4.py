"""NumPy-based executor for dense GLM-4 architecture (Glm4ForCausalLM)."""
from __future__ import annotations

import math
from typing import Any

from offload_runtime.backends.base import DeviceBackend
from offload_runtime.types import DeviceBuffer, LayerSpec

from ._common import (
    _readback_device, _unpack_tensors, linear_t, np, repeat_kv, rms_norm, rope,
    silu, softmax,
)

LAYER_TENSORS = [
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


class GLM4Executor:
    """NumPy-based executor for dense GLM-4 architecture.

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
