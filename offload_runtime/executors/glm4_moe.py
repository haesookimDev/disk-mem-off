"""NumPy-based executor for GLM-4 MoE architecture (Glm4MoeForCausalLM).

Supports zai-org/GLM-4.7 and similar models with:
- Hybrid dense/MoE layers (first_k_dense_replace dense, rest MoE)
- Grouped Query Attention with QK normalization
- Partial Rotary Position Embeddings
- Top-K expert routing with shared expert
"""
from __future__ import annotations

import math
from typing import Any

from offload_runtime.backends.base import DeviceBackend
from offload_runtime.types import DeviceBuffer, LayerSpec

from ._common import (
    _readback_device, _unpack_tensors, linear_t, np, repeat_kv, rms_norm, rope,
    silu, softmax,
)

# Dense layers (first_k_dense_replace) use these tensors.
DENSE_LAYER_TENSORS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.q_proj.bias",
    "self_attn.k_proj.weight", "self_attn.k_proj.bias",
    "self_attn.v_proj.weight", "self_attn.v_proj.bias",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

# MoE layers use attention tensors + router + experts (dynamic).
# The expert tensors are collected dynamically by the loader.
MOE_ATTN_TENSORS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.q_proj.bias",
    "self_attn.k_proj.weight", "self_attn.k_proj.bias",
    "self_attn.v_proj.weight", "self_attn.v_proj.bias",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
]

MOE_ROUTER_TENSORS = [
    "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias",
]

MOE_SHARED_EXPERT_TENSORS = [
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
]

# Per-expert tensors: mlp.experts.{i}.{gate,up,down}_proj.weight
_EXPERT_TENSOR_SUFFIXES = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]


def _rms_norm_no_weight(x: Any, eps: float = 1e-6) -> Any:
    """RMS normalization without learnable weight (for QK norm)."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms


def _partial_rope(
    q: Any, k: Any, positions: Any, head_dim: int,
    rotary_dim: int, base: float,
) -> tuple[Any, Any]:
    """Apply RoPE only to the first rotary_dim dimensions."""
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot, k_rot = rope(q_rot, k_rot, positions, rotary_dim, base)
    return (
        np.concatenate([q_rot, q_pass], axis=-1),
        np.concatenate([k_rot, k_pass], axis=-1),
    )


class GLM4MoeExecutor:
    """NumPy-based executor for GLM-4 MoE architecture (Glm4MoeForCausalLM).

    Handles both dense layers (first_k_dense_replace) and MoE layers,
    dispatched via layer.metadata["is_moe"].
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.hidden_size: int = config["hidden_size"]
        self.num_heads: int = config["num_attention_heads"]
        self.num_kv_heads: int = config.get("num_key_value_heads", self.num_heads)
        self.head_dim: int = config.get("head_dim", self.hidden_size // self.num_heads)
        self.intermediate_size: int = config["intermediate_size"]
        self.moe_intermediate_size: int = config.get("moe_intermediate_size", self.intermediate_size)
        self.rms_norm_eps: float = config.get("rms_norm_eps", 1e-5)
        self.rope_theta: float = config.get("rope_theta", 1000000.0)
        self.partial_rotary_factor: float = config.get("partial_rotary_factor", 0.5)
        self.rotary_dim: int = int(self.head_dim * self.partial_rotary_factor)
        self.num_experts_per_tok: int = config.get("num_experts_per_tok", 8)
        self.n_routed_experts: int = config.get("n_routed_experts", 160)
        self.routed_scaling_factor: float = config.get("routed_scaling_factor", 2.5)
        self.norm_topk_prob: bool = config.get("norm_topk_prob", True)

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
        x = self._attention_block(x, t)

        is_moe = layer.metadata.get("is_moe", False)
        if is_moe:
            x = self._moe_mlp_block(x, t)
        else:
            x = self._dense_mlp_block(x, t)

        return x

    # -- Attention (shared between dense and MoE layers) --

    def _attention_block(self, x: Any, t: dict[str, Any]) -> Any:
        seq_len = x.shape[0]
        n_heads = self.num_heads
        n_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

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

        # Reshape to [n_heads, seq_len, head_dim]
        q = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)

        # QK normalization (per-head RMS norm without weight, then scale)
        if "self_attn.q_norm.weight" in t:
            q = _rms_norm_no_weight(q, self.rms_norm_eps) * t["self_attn.q_norm.weight"]
        if "self_attn.k_norm.weight" in t:
            k = _rms_norm_no_weight(k, self.rms_norm_eps) * t["self_attn.k_norm.weight"]

        # Partial RoPE
        positions = np.arange(seq_len, dtype=np.float32)
        if self.rotary_dim < head_dim:
            q, k = _partial_rope(q, k, positions, head_dim, self.rotary_dim, self.rope_theta)
        else:
            q, k = rope(q, k, positions, head_dim, self.rope_theta)

        # GQA repeat
        n_rep = n_heads // n_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_dim)
        mask = np.triu(np.full((seq_len, seq_len), -1e10, dtype=np.float32), k=1)
        scores = scores + mask
        attn_w = softmax(scores, axis=-1)
        attn_out = attn_w @ v

        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)
        attn_out = linear_t(attn_out, t["self_attn.o_proj.weight"])
        return x + attn_out

    # -- Dense MLP (for first_k_dense_replace layers) --

    def _dense_mlp_block(self, x: Any, t: dict[str, Any]) -> Any:
        h = rms_norm(x, t["post_attention_layernorm.weight"], self.rms_norm_eps)
        gate = linear_t(h, t["mlp.gate_proj.weight"])
        up = linear_t(h, t["mlp.up_proj.weight"])
        h = silu(gate) * up
        h = linear_t(h, t["mlp.down_proj.weight"])
        return x + h

    # -- MoE MLP (for remaining layers) --

    def _moe_mlp_block(self, x: Any, t: dict[str, Any]) -> Any:
        h = rms_norm(x, t["post_attention_layernorm.weight"], self.rms_norm_eps)
        seq_len = h.shape[0]

        # --- Router ---
        router_logits = h @ t["mlp.gate.weight"].T  # [seq_len, n_experts]
        if "mlp.gate.e_score_correction_bias" in t:
            router_logits = router_logits + t["mlp.gate.e_score_correction_bias"]

        router_probs = softmax(router_logits, axis=-1)

        # Top-K selection
        k = self.num_experts_per_tok
        topk_indices = np.argpartition(-router_probs, k, axis=-1)[:, :k]
        topk_weights = np.take_along_axis(router_probs, topk_indices, axis=-1)

        # Sort within top-k for deterministic ordering
        sort_idx = np.argsort(-topk_weights, axis=-1)
        topk_indices = np.take_along_axis(topk_indices, sort_idx, axis=-1)
        topk_weights = np.take_along_axis(topk_weights, sort_idx, axis=-1)

        # Normalize and scale
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(axis=-1, keepdims=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor

        # --- Expert computation ---
        output = np.zeros_like(h)

        # Collect unique experts that need computation
        active_experts = set(topk_indices.ravel().tolist())

        for expert_idx in active_experts:
            # Which tokens route to this expert, and their weights
            mask = (topk_indices == expert_idx)  # [seq_len, k]
            token_mask = mask.any(axis=-1)  # [seq_len]
            if not token_mask.any():
                continue

            expert_input = h[token_mask]  # [n_tokens, hidden]

            # SwiGLU for this expert
            gate_w = t[f"mlp.experts.{expert_idx}.gate_proj.weight"]
            up_w = t[f"mlp.experts.{expert_idx}.up_proj.weight"]
            down_w = t[f"mlp.experts.{expert_idx}.down_proj.weight"]

            gate_out = silu(linear_t(expert_input, gate_w))
            up_out = linear_t(expert_input, up_w)
            expert_out = linear_t(gate_out * up_out, down_w)  # [n_tokens, hidden]

            # Scatter weighted expert output back
            token_indices = np.where(token_mask)[0]
            for local_i, tok_i in enumerate(token_indices):
                # Sum weights for this expert across all top-k slots
                w = topk_weights[tok_i][mask[tok_i]].sum()
                output[tok_i] += w * expert_out[local_i]

        # --- Shared expert (always active) ---
        if "mlp.shared_experts.gate_proj.weight" in t:
            shared_gate = silu(linear_t(h, t["mlp.shared_experts.gate_proj.weight"]))
            shared_up = linear_t(h, t["mlp.shared_experts.up_proj.weight"])
            shared_out = linear_t(shared_gate * shared_up, t["mlp.shared_experts.down_proj.weight"])
            output = output + shared_out

        return x + output

    # -- Non-layer helpers --

    def embed(self, token_ids: list[int], embed_weights: Any) -> Any:
        return embed_weights[token_ids]

    def lm_head(self, hidden: Any, norm_w: Any, head_w: Any) -> Any:
        h = rms_norm(hidden, norm_w, self.rms_norm_eps)
        return h @ head_w.T
