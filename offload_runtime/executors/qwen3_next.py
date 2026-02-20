"""NumPy-based executor for Qwen3-Coder-Next (Qwen3NextForCausalLM).

Hybrid architecture with:
- Gated DeltaNet linear attention (most layers)
- Gated Full GQA attention (every full_attention_interval-th layer)
- 512-expert MoE with shared expert on ALL layers
"""
from __future__ import annotations

import math
from typing import Any

from offload_runtime.backends.base import DeviceBackend
from offload_runtime.types import DeviceBuffer, LayerSpec

from ._common import (
    _ensure_f32, _readback_device, _unpack_tensors, linear_t, np, partial_rope,
    repeat_kv, rms_norm, rms_norm_no_weight, rope, silu, softmax,
)


# ---------------------------------------------------------------------------
# Gated DeltaNet helpers
# ---------------------------------------------------------------------------

def _softplus(x: Any) -> Any:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def _sigmoid(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


def _causal_conv1d(x: Any, weight: Any) -> Any:
    """Depthwise causal 1D convolution.

    x: [seq_len, channels]
    weight: [channels, kernel_size]  (depthwise — each channel has its own kernel)
    returns: [seq_len, channels]
    """
    seq_len, channels = x.shape
    kernel_size = weight.shape[-1]

    # Left-pad with zeros for causal behavior
    padded = np.zeros((seq_len + kernel_size - 1, channels), dtype=x.dtype)
    padded[kernel_size - 1:] = x

    out = np.zeros_like(x)
    for k in range(kernel_size):
        out += padded[kernel_size - 1 - k: seq_len + kernel_size - 1 - k] * weight[:, k]
    return out


def _gated_delta_rule_recurrence(
    q: Any, k: Any, v: Any, g: Any, beta: Any,
    num_heads: int, head_dim_k: int, head_dim_v: int,
) -> Any:
    """Gated DeltaNet linear recurrence.

    q:    [seq_len, num_heads * head_dim_k]
    k:    [seq_len, num_heads * head_dim_k]
    v:    [seq_len, num_heads * head_dim_v]
    g:    [seq_len, num_heads]  (per-head decay, negative log-space)
    beta: [seq_len, num_heads * head_dim_k]  (input gate after sigmoid)

    State recurrence (per head):
        S_t = exp(g_t) * S_{t-1} + (beta_t * k_t)^T @ v_t
        o_t = S_t @ q_t
    """
    seq_len = q.shape[0]
    q = q.reshape(seq_len, num_heads, head_dim_k)
    k = k.reshape(seq_len, num_heads, head_dim_k)
    v = v.reshape(seq_len, num_heads, head_dim_v)
    beta = beta.reshape(seq_len, num_heads, head_dim_k)

    # State: [num_heads, head_dim_v, head_dim_k]
    S = np.zeros((num_heads, head_dim_v, head_dim_k), dtype=q.dtype)
    output = np.zeros((seq_len, num_heads, head_dim_v), dtype=q.dtype)

    for t_idx in range(seq_len):
        # Decay factor: exp(g_t) per head → broadcast to state shape
        decay = np.exp(g[t_idx])  # [num_heads]
        S = S * decay[:, None, None]

        # Update: outer product of v_t and (beta_t * k_t)
        bk = beta[t_idx] * k[t_idx]  # [num_heads, head_dim_k]
        # v_t: [num_heads, head_dim_v], bk: [num_heads, head_dim_k]
        # outer per head: [num_heads, head_dim_v, head_dim_k]
        S = S + v[t_idx, :, :, None] * bk[:, None, :]

        # Output: S @ q_t → [num_heads, head_dim_v]
        output[t_idx] = np.einsum("hvk,hk->hv", S, q[t_idx])

    return output.reshape(seq_len, num_heads * head_dim_v)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Qwen3NextExecutor:
    """NumPy executor for Qwen3NextForCausalLM (Qwen3-Coder-Next)."""

    def __init__(self, config: dict[str, Any]) -> None:
        # Common
        self.hidden_size: int = config["hidden_size"]
        self.rms_norm_eps: float = config.get("rms_norm_eps", 1e-6)
        self.rope_theta: float = config.get("rope_theta", 5000000.0)
        self.full_attention_interval: int = config.get("full_attention_interval", 4)

        # Full attention
        self.num_heads: int = config["num_attention_heads"]
        self.num_kv_heads: int = config.get("num_key_value_heads", self.num_heads)
        self.head_dim: int = config.get("head_dim", self.hidden_size // self.num_heads)
        self.partial_rotary_factor: float = config.get("partial_rotary_factor", 0.25)
        self.rotary_dim: int = int(self.head_dim * self.partial_rotary_factor)

        # Linear attention (Gated DeltaNet)
        self.linear_num_key_heads: int = config.get("linear_num_key_heads", 16)
        self.linear_key_head_dim: int = config.get("linear_key_head_dim", 128)
        self.linear_num_value_heads: int = config.get("linear_num_value_heads", 32)
        self.linear_value_head_dim: int = config.get("linear_value_head_dim", 128)
        self.linear_conv_kernel_dim: int = config.get("linear_conv_kernel_dim", 4)

        # MoE
        self.num_experts: int = config.get("num_experts", config.get("n_routed_experts", 512))
        self.num_experts_per_tok: int = config.get("num_experts_per_tok", 10)
        self.moe_intermediate_size: int = config.get("moe_intermediate_size", 512)
        self.shared_expert_intermediate_size: int = config.get(
            "shared_expert_intermediate_size", self.moe_intermediate_size,
        )
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
        attn_type = layer.metadata.get("attn_type", "linear")

        if attn_type == "full":
            x = self._full_attn_block(x, t)
        else:
            x = self._linear_attn_block(x, t)

        x = self._moe_mlp_block(x, t)
        return x

    # -- Full Attention (Gated GQA) --

    def _full_attn_block(self, x: Any, t: dict[str, Any]) -> Any:
        seq_len = x.shape[0]
        n_heads = self.num_heads
        n_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        h = rms_norm(x, t["input_layernorm.weight"], self.rms_norm_eps)

        # q_proj outputs 2x: [query, gate]
        qg = linear_t(h, t["self_attn.q_proj.weight"])  # [seq, n_heads * head_dim * 2]
        q, gate = np.split(qg, 2, axis=-1)

        k = linear_t(h, t["self_attn.k_proj.weight"])
        v = linear_t(h, t["self_attn.v_proj.weight"])

        # Reshape to [n_heads, seq_len, head_dim]
        q = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)

        # QK normalization
        if "self_attn.q_norm.weight" in t:
            q = rms_norm_no_weight(q, self.rms_norm_eps) * t["self_attn.q_norm.weight"]
        if "self_attn.k_norm.weight" in t:
            k = rms_norm_no_weight(k, self.rms_norm_eps) * t["self_attn.k_norm.weight"]

        # Partial RoPE
        positions = np.arange(seq_len, dtype=np.float32)
        if self.rotary_dim < head_dim:
            q, k = partial_rope(q, k, positions, head_dim, self.rotary_dim, self.rope_theta)
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

        # Gated output: attn_out * sigmoid(gate)
        attn_out = attn_out * _sigmoid(gate)

        attn_out = linear_t(attn_out, t["self_attn.o_proj.weight"])
        return x + attn_out

    # -- Linear Attention (Gated DeltaNet) --

    def _linear_attn_block(self, x: Any, t: dict[str, Any]) -> Any:
        seq_len = x.shape[0]
        num_k_heads = self.linear_num_key_heads
        k_head_dim = self.linear_key_head_dim
        num_v_heads = self.linear_num_value_heads
        v_head_dim = self.linear_value_head_dim

        q_dim = num_k_heads * k_head_dim
        k_dim = num_k_heads * k_head_dim
        v_dim = num_v_heads * v_head_dim
        z_dim = num_v_heads * v_head_dim  # same as v_dim for gating

        h = rms_norm(x, t["input_layernorm.weight"], self.rms_norm_eps)

        # Project Q, K, V, Z
        qkvz = linear_t(h, t["linear_attn.in_proj_qkvz.weight"])  # [seq, q+k+v+z]
        q_raw, k_raw, v_raw, z = np.split(qkvz, [q_dim, q_dim + k_dim, q_dim + k_dim + v_dim], axis=-1)

        # Causal conv1d on concat(Q, K, V) then SiLU
        qkv_cat = np.concatenate([q_raw, k_raw, v_raw], axis=-1)  # [seq, q+k+v]
        conv_w = t["linear_attn.conv1d.weight"]
        # conv1d weight may be [channels, 1, kernel] or [channels, kernel]
        if conv_w.ndim == 3:
            conv_w = conv_w.squeeze(1)  # [channels, kernel]
        qkv_conv = _causal_conv1d(qkv_cat, conv_w)
        qkv_conv = silu(qkv_conv)
        q, k, v = np.split(qkv_conv, [q_dim, q_dim + k_dim], axis=-1)

        # Project B (input gate) and A (decay input)
        ba = linear_t(h, t["linear_attn.in_proj_ba.weight"])  # [seq, b_dim + a_dim]
        b_dim = num_k_heads * k_head_dim
        beta_raw, a_raw = np.split(ba, [b_dim], axis=-1)
        beta = _sigmoid(beta_raw)  # input gate

        # Compute decay: g = -exp(A_log) * softplus(a + dt_bias)
        A_log = t["linear_attn.A_log"]   # [num_v_heads]
        dt_bias = t["linear_attn.dt_bias"]  # [num_v_heads]
        # a_raw: [seq, num_v_heads]
        g = -np.exp(A_log) * _softplus(a_raw + dt_bias)  # [seq, num_v_heads]

        # Gated DeltaNet recurrence
        output = _gated_delta_rule_recurrence(
            q, k, v, g, beta,
            num_v_heads, k_head_dim, v_head_dim,
        )  # [seq, v_dim]

        # Gated RMS norm: rms_norm(output) * silu(z)
        norm_w = t["linear_attn.norm.weight"]  # [v_dim]
        output = rms_norm(output, norm_w, self.rms_norm_eps)
        output = output * silu(z)

        # Output projection
        output = linear_t(output, t["linear_attn.out_proj.weight"])
        return x + output

    # -- MoE MLP (all layers) --

    def _moe_mlp_block(self, x: Any, t: dict[str, Any]) -> Any:
        h = rms_norm(x, t["post_attention_layernorm.weight"], self.rms_norm_eps)
        seq_len = h.shape[0]

        # --- Router ---
        router_logits = h @ t["mlp.gate.weight"].T  # [seq_len, n_experts]
        router_probs = softmax(router_logits, axis=-1)

        # Top-K selection
        k = self.num_experts_per_tok
        topk_indices = np.argpartition(-router_probs, k, axis=-1)[:, :k]
        topk_weights = np.take_along_axis(router_probs, topk_indices, axis=-1)

        # Sort within top-k for deterministic ordering
        sort_idx = np.argsort(-topk_weights, axis=-1)
        topk_indices = np.take_along_axis(topk_indices, sort_idx, axis=-1)
        topk_weights = np.take_along_axis(topk_weights, sort_idx, axis=-1)

        # Normalize
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(axis=-1, keepdims=True) + 1e-20)

        # --- Expert computation ---
        output = np.zeros_like(h)
        active_experts = set(topk_indices.ravel().tolist())

        for expert_idx in active_experts:
            mask = (topk_indices == expert_idx)  # [seq_len, k]
            token_mask = mask.any(axis=-1)  # [seq_len]
            if not token_mask.any():
                continue

            expert_input = h[token_mask]

            gate_w = t[f"mlp.experts.{expert_idx}.gate_proj.weight"]
            up_w = t[f"mlp.experts.{expert_idx}.up_proj.weight"]
            down_w = t[f"mlp.experts.{expert_idx}.down_proj.weight"]

            gate_out = silu(linear_t(expert_input, gate_w))
            up_out = linear_t(expert_input, up_w)
            expert_out = linear_t(gate_out * up_out, down_w)

            token_indices = np.where(token_mask)[0]
            for local_i, tok_i in enumerate(token_indices):
                w = topk_weights[tok_i][mask[tok_i]].sum()
                output[tok_i] += w * expert_out[local_i]

        # --- Shared expert with sigmoid gate ---
        if "mlp.shared_expert.gate_proj.weight" in t:
            shared_gate = silu(linear_t(h, t["mlp.shared_expert.gate_proj.weight"]))
            shared_up = linear_t(h, t["mlp.shared_expert.up_proj.weight"])
            shared_out = linear_t(shared_gate * shared_up, t["mlp.shared_expert.down_proj.weight"])

            # Apply learned sigmoid gate if present
            if "mlp.shared_expert_gate.weight" in t:
                gate_val = _sigmoid(h @ t["mlp.shared_expert_gate.weight"].T)  # [seq, 1]
                shared_out = shared_out * gate_val

            output = output + shared_out

        return x + output

    # -- Non-layer helpers --

    def embed(self, token_ids: list[int], embed_weights: Any) -> Any:
        return _ensure_f32(embed_weights[token_ids])

    def lm_head(self, hidden: Any, norm_w: Any, head_w: Any) -> Any:
        h = rms_norm(hidden, norm_w, self.rms_norm_eps)
        return h @ head_w.T
