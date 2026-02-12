from __future__ import annotations

import struct
from typing import Any

import pytest

try:
    import numpy as np

    _has_numpy = True
except Exception:
    _has_numpy = False

pytestmark = pytest.mark.skipif(not _has_numpy, reason="numpy not installed")


# ---------------------------------------------------------------------------
# Imports guarded behind numpy availability
# ---------------------------------------------------------------------------
if _has_numpy:
    from offload_runtime.backends.null_backend import NullBackend
    from offload_runtime.executor_np import (
        GPT2Executor,
        LlamaExecutor,
        _readback_device,
        _unpack_tensors,
        gelu,
        layer_norm,
        linear,
        linear_t,
        repeat_kv,
        rms_norm,
        rope,
        silu,
        softmax,
    )
    from offload_runtime.runtime import OffloadRuntime
    from offload_runtime.scheduler.lookahead import LookaheadScheduler
    from offload_runtime.storage.in_memory import InMemoryStorage
    from offload_runtime.types import DeviceBuffer, HostBuffer, LayerSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack_tensors(
    tensors: dict[str, Any],
    order: list[str],
    dtype: str = "float32",
) -> tuple[bytes, list[dict[str, Any]]]:
    """Pack named numpy arrays into contiguous bytes + metadata list."""
    meta: list[dict[str, Any]] = []
    parts: list[bytes] = []
    offset = 0
    for name in order:
        arr = tensors[name].astype(dtype)
        raw = arr.tobytes()
        meta.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": dtype,
            "offset": offset,
            "nbytes": len(raw),
        })
        parts.append(raw)
        offset += len(raw)
    return b"".join(parts), meta


def _random_gpt2_layer_weights(
    n_embd: int, rng: Any,
) -> dict[str, Any]:
    return {
        "ln_1.weight": rng.standard_normal(n_embd).astype(np.float32),
        "ln_1.bias": rng.standard_normal(n_embd).astype(np.float32),
        "attn.c_attn.weight": (rng.standard_normal((n_embd, 3 * n_embd)) * 0.02).astype(np.float32),
        "attn.c_attn.bias": np.zeros(3 * n_embd, dtype=np.float32),
        "attn.c_proj.weight": (rng.standard_normal((n_embd, n_embd)) * 0.02).astype(np.float32),
        "attn.c_proj.bias": np.zeros(n_embd, dtype=np.float32),
        "ln_2.weight": rng.standard_normal(n_embd).astype(np.float32),
        "ln_2.bias": rng.standard_normal(n_embd).astype(np.float32),
        "mlp.c_fc.weight": (rng.standard_normal((n_embd, 4 * n_embd)) * 0.02).astype(np.float32),
        "mlp.c_fc.bias": np.zeros(4 * n_embd, dtype=np.float32),
        "mlp.c_proj.weight": (rng.standard_normal((4 * n_embd, n_embd)) * 0.02).astype(np.float32),
        "mlp.c_proj.bias": np.zeros(n_embd, dtype=np.float32),
    }


def _random_llama_layer_weights(
    hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int, rng: Any,
) -> dict[str, Any]:
    head_dim = hidden_size // num_heads
    return {
        "input_layernorm.weight": np.ones(hidden_size, dtype=np.float32),
        "self_attn.q_proj.weight": (rng.standard_normal((hidden_size, hidden_size)) * 0.02).astype(np.float32),
        "self_attn.k_proj.weight": (rng.standard_normal((num_kv_heads * head_dim, hidden_size)) * 0.02).astype(np.float32),
        "self_attn.v_proj.weight": (rng.standard_normal((num_kv_heads * head_dim, hidden_size)) * 0.02).astype(np.float32),
        "self_attn.o_proj.weight": (rng.standard_normal((hidden_size, hidden_size)) * 0.02).astype(np.float32),
        "post_attention_layernorm.weight": np.ones(hidden_size, dtype=np.float32),
        "mlp.gate_proj.weight": (rng.standard_normal((intermediate_size, hidden_size)) * 0.02).astype(np.float32),
        "mlp.up_proj.weight": (rng.standard_normal((intermediate_size, hidden_size)) * 0.02).astype(np.float32),
        "mlp.down_proj.weight": (rng.standard_normal((hidden_size, intermediate_size)) * 0.02).astype(np.float32),
    }


_GPT2_ORDER = [
    "ln_1.weight", "ln_1.bias",
    "attn.c_attn.weight", "attn.c_attn.bias",
    "attn.c_proj.weight", "attn.c_proj.bias",
    "ln_2.weight", "ln_2.bias",
    "mlp.c_fc.weight", "mlp.c_fc.bias",
    "mlp.c_proj.weight", "mlp.c_proj.bias",
]

_LLAMA_ORDER = [
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


# ---------------------------------------------------------------------------
# Math function tests
# ---------------------------------------------------------------------------

class TestMathFunctions:
    def test_layer_norm_identity(self) -> None:
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        out = layer_norm(x, w, b)
        assert out.shape == (1, 4)
        assert np.abs(out.mean()) < 1e-5
        assert np.abs(out.var() - 1.0) < 0.1

    def test_layer_norm_with_params(self) -> None:
        x = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        w = np.ones(4, dtype=np.float32) * 2.0
        b = np.ones(4, dtype=np.float32) * 3.0
        out = layer_norm(x, w, b)
        np.testing.assert_allclose(out, 3.0, atol=1e-5)

    def test_rms_norm_identity(self) -> None:
        x = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        out = rms_norm(x, w)
        np.testing.assert_allclose(out, 1.0, atol=1e-5)

    def test_rms_norm_scaling(self) -> None:
        x = np.array([[2.0, 2.0]], dtype=np.float32)
        w = np.ones(2, dtype=np.float32)
        out = rms_norm(x, w)
        np.testing.assert_allclose(out, 1.0, atol=1e-5)

    def test_gelu_zero(self) -> None:
        assert abs(float(gelu(np.array(0.0)))) < 1e-6

    def test_gelu_positive(self) -> None:
        out = float(gelu(np.array(3.0)))
        assert out > 2.9

    def test_gelu_negative(self) -> None:
        out = float(gelu(np.array(-5.0)))
        assert abs(out) < 0.01

    def test_silu_zero(self) -> None:
        assert abs(float(silu(np.array(0.0)))) < 1e-6

    def test_silu_positive(self) -> None:
        out = float(silu(np.array(5.0)))
        assert out > 4.9

    def test_softmax_sums_to_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = softmax(x)
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-6)

    def test_softmax_max_gets_largest(self) -> None:
        x = np.array([1.0, 10.0, 2.0], dtype=np.float32)
        out = softmax(x)
        assert np.argmax(out) == 1

    def test_linear_with_bias(self) -> None:
        x = np.array([[1.0, 0.0]], dtype=np.float32)
        w = np.array([[2.0], [3.0]], dtype=np.float32)
        b = np.array([1.0], dtype=np.float32)
        out = linear(x, w, b)
        np.testing.assert_allclose(out, [[3.0]])

    def test_linear_t(self) -> None:
        x = np.array([[1.0, 0.0]], dtype=np.float32)
        w = np.array([[2.0, 3.0]], dtype=np.float32)  # [out=1, in=2]
        out = linear_t(x, w)
        np.testing.assert_allclose(out, [[2.0]])

    def test_rope_preserves_shape(self) -> None:
        rng = np.random.default_rng(42)
        q = rng.standard_normal((4, 3, 8)).astype(np.float32)  # [n_heads, seq, head_dim]
        k = rng.standard_normal((4, 3, 8)).astype(np.float32)
        positions = np.arange(3, dtype=np.float32)
        q_out, k_out = rope(q, k, positions, head_dim=8)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_repeat_kv_noop(self) -> None:
        x = np.ones((2, 3, 4), dtype=np.float32)
        out = repeat_kv(x, 1)
        assert out is x

    def test_repeat_kv_doubles(self) -> None:
        x = np.ones((2, 3, 4), dtype=np.float32)
        out = repeat_kv(x, 2)
        assert out.shape == (4, 3, 4)


# ---------------------------------------------------------------------------
# GPT-2 Executor tests
# ---------------------------------------------------------------------------

class TestGPT2Executor:
    def setup_method(self) -> None:
        self.n_embd = 8
        self.n_head = 2
        self.config = {"n_embd": self.n_embd, "n_head": self.n_head, "n_positions": 16}
        self.executor = GPT2Executor(self.config)
        self.rng = np.random.default_rng(42)

    def _make_layer(self) -> tuple[LayerSpec, bytes]:
        weights = _random_gpt2_layer_weights(self.n_embd, self.rng)
        raw, meta = _pack_tensors(weights, _GPT2_ORDER)
        spec = LayerSpec(layer_id=0, name="h.0", nbytes=len(raw), metadata={"tensors": meta})
        return spec, raw

    def test_run_layer_output_shape(self) -> None:
        spec, raw = self._make_layer()
        backend = NullBackend()
        stream = backend.create_stream("test")
        device_buf = backend.alloc_device(len(raw))
        src = HostBuffer(view=memoryview(bytearray(raw)), pinned=False)
        backend.copy_h2d_async(device_buf, src, stream)

        x = self.rng.standard_normal((3, self.n_embd)).astype(np.float32)
        out = self.executor.run_layer(spec, x, device_buf, backend, stream)
        assert out.shape == (3, self.n_embd)
        backend.free_device(device_buf)

    def test_run_layer_changes_activations(self) -> None:
        spec, raw = self._make_layer()
        backend = NullBackend()
        stream = backend.create_stream("test")
        device_buf = backend.alloc_device(len(raw))
        src = HostBuffer(view=memoryview(bytearray(raw)), pinned=False)
        backend.copy_h2d_async(device_buf, src, stream)

        x = self.rng.standard_normal((3, self.n_embd)).astype(np.float32)
        out = self.executor.run_layer(spec, x, device_buf, backend, stream)
        assert not np.allclose(out, x)
        backend.free_device(device_buf)

    def test_embed_shape(self) -> None:
        wte = self.rng.standard_normal((16, self.n_embd)).astype(np.float32)
        wpe = self.rng.standard_normal((16, self.n_embd)).astype(np.float32)
        out = self.executor.embed([0, 1, 2], wte, wpe)
        assert out.shape == (3, self.n_embd)

    def test_lm_head_shape(self) -> None:
        hidden = self.rng.standard_normal((3, self.n_embd)).astype(np.float32)
        ln_f_w = np.ones(self.n_embd, dtype=np.float32)
        ln_f_b = np.zeros(self.n_embd, dtype=np.float32)
        head_w = self.rng.standard_normal((16, self.n_embd)).astype(np.float32)
        out = self.executor.lm_head(hidden, ln_f_w, ln_f_b, head_w)
        assert out.shape == (3, 16)

    def test_runtime_integration(self) -> None:
        """GPT2Executor works with OffloadRuntime end-to-end."""
        spec, raw = self._make_layer()
        storage = InMemoryStorage({0: raw})
        backend = NullBackend()

        with OffloadRuntime(
            layers=[spec],
            backend=backend,
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=self.executor,
        ) as runtime:
            x = self.rng.standard_normal((2, self.n_embd)).astype(np.float32)
            out, metrics = runtime.run_inference([0], inputs=x)

        assert out.shape == (2, self.n_embd)
        assert metrics.layer_count == 1
        assert metrics.transferred_bytes == len(raw)


# ---------------------------------------------------------------------------
# LLaMA Executor tests
# ---------------------------------------------------------------------------

class TestLlamaExecutor:
    def setup_method(self) -> None:
        self.hidden_size = 8
        self.num_heads = 2
        self.num_kv_heads = 1
        self.intermediate_size = 16
        self.config = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
        }
        self.executor = LlamaExecutor(self.config)
        self.rng = np.random.default_rng(123)

    def _make_layer(self) -> tuple[LayerSpec, bytes]:
        weights = _random_llama_layer_weights(
            self.hidden_size, self.num_heads, self.num_kv_heads,
            self.intermediate_size, self.rng,
        )
        raw, meta = _pack_tensors(weights, _LLAMA_ORDER)
        spec = LayerSpec(layer_id=0, name="layers.0", nbytes=len(raw), metadata={"tensors": meta})
        return spec, raw

    def test_run_layer_output_shape(self) -> None:
        spec, raw = self._make_layer()
        backend = NullBackend()
        stream = backend.create_stream("test")
        device_buf = backend.alloc_device(len(raw))
        src = HostBuffer(view=memoryview(bytearray(raw)), pinned=False)
        backend.copy_h2d_async(device_buf, src, stream)

        x = self.rng.standard_normal((3, self.hidden_size)).astype(np.float32)
        out = self.executor.run_layer(spec, x, device_buf, backend, stream)
        assert out.shape == (3, self.hidden_size)
        backend.free_device(device_buf)

    def test_run_layer_changes_activations(self) -> None:
        spec, raw = self._make_layer()
        backend = NullBackend()
        stream = backend.create_stream("test")
        device_buf = backend.alloc_device(len(raw))
        src = HostBuffer(view=memoryview(bytearray(raw)), pinned=False)
        backend.copy_h2d_async(device_buf, src, stream)

        x = self.rng.standard_normal((3, self.hidden_size)).astype(np.float32)
        out = self.executor.run_layer(spec, x, device_buf, backend, stream)
        assert not np.allclose(out, x)
        backend.free_device(device_buf)

    def test_embed_shape(self) -> None:
        embed_w = self.rng.standard_normal((16, self.hidden_size)).astype(np.float32)
        out = self.executor.embed([0, 1, 2], embed_w)
        assert out.shape == (3, self.hidden_size)

    def test_lm_head_shape(self) -> None:
        hidden = self.rng.standard_normal((3, self.hidden_size)).astype(np.float32)
        norm_w = np.ones(self.hidden_size, dtype=np.float32)
        head_w = self.rng.standard_normal((16, self.hidden_size)).astype(np.float32)
        out = self.executor.lm_head(hidden, norm_w, head_w)
        assert out.shape == (3, 16)

    def test_runtime_integration(self) -> None:
        """LlamaExecutor works with OffloadRuntime end-to-end."""
        spec, raw = self._make_layer()
        storage = InMemoryStorage({0: raw})
        backend = NullBackend()

        with OffloadRuntime(
            layers=[spec],
            backend=backend,
            storage=storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=self.executor,
        ) as runtime:
            x = self.rng.standard_normal((2, self.hidden_size)).astype(np.float32)
            out, metrics = runtime.run_inference([0], inputs=x)

        assert out.shape == (2, self.hidden_size)
        assert metrics.layer_count == 1


# ---------------------------------------------------------------------------
# Unpack/readback helpers tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_unpack_tensors_roundtrip(self) -> None:
        w1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        w2 = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
        raw, meta = _pack_tensors({"a": w1, "b": w2}, ["a", "b"])
        result = _unpack_tensors(raw, meta)
        np.testing.assert_array_equal(result["a"], w1)
        np.testing.assert_array_equal(result["b"], w2)

    def test_readback_device(self) -> None:
        backend = NullBackend()
        stream = backend.create_stream("test")
        data = b"\x01\x02\x03\x04"
        device_buf = backend.alloc_device(4)
        src = HostBuffer(view=memoryview(bytearray(data)), pinned=False)
        backend.copy_h2d_async(device_buf, src, stream)
        result = _readback_device(device_buf, backend, stream)
        assert result == data
        backend.free_device(device_buf)
