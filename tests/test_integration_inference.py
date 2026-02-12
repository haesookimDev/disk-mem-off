from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

try:
    import numpy as np
    from safetensors.numpy import save_file

    _has_deps = True
except Exception:
    _has_deps = False

pytestmark = pytest.mark.skipif(not _has_deps, reason="inference deps not installed")

if _has_deps:
    from offload_runtime.backends.null_backend import NullBackend
    from offload_runtime.executor_np import GPT2Executor, LlamaExecutor
    from offload_runtime.loader.huggingface import HuggingFaceLoader
    from offload_runtime.runtime import OffloadRuntime
    from offload_runtime.scheduler.lookahead import LookaheadScheduler


def _make_gpt2_model(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    n_embd, n_head, n_layer, vocab = 8, 2, 2, 16
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
        "n_positions": 16, "vocab_size": vocab,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    tensors: dict[str, Any] = {
        "transformer.wte.weight": rng.standard_normal((vocab, n_embd)).astype(np.float32),
        "transformer.wpe.weight": rng.standard_normal((16, n_embd)).astype(np.float32),
        "transformer.ln_f.weight": np.ones(n_embd, dtype=np.float32),
        "transformer.ln_f.bias": np.zeros(n_embd, dtype=np.float32),
    }
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        tensors[f"{p}ln_1.weight"] = np.ones(n_embd, dtype=np.float32)
        tensors[f"{p}ln_1.bias"] = np.zeros(n_embd, dtype=np.float32)
        tensors[f"{p}attn.c_attn.weight"] = (rng.standard_normal((n_embd, 3 * n_embd)) * 0.02).astype(np.float32)
        tensors[f"{p}attn.c_attn.bias"] = np.zeros(3 * n_embd, dtype=np.float32)
        tensors[f"{p}attn.c_proj.weight"] = (rng.standard_normal((n_embd, n_embd)) * 0.02).astype(np.float32)
        tensors[f"{p}attn.c_proj.bias"] = np.zeros(n_embd, dtype=np.float32)
        tensors[f"{p}ln_2.weight"] = np.ones(n_embd, dtype=np.float32)
        tensors[f"{p}ln_2.bias"] = np.zeros(n_embd, dtype=np.float32)
        tensors[f"{p}mlp.c_fc.weight"] = (rng.standard_normal((n_embd, 4 * n_embd)) * 0.02).astype(np.float32)
        tensors[f"{p}mlp.c_fc.bias"] = np.zeros(4 * n_embd, dtype=np.float32)
        tensors[f"{p}mlp.c_proj.weight"] = (rng.standard_normal((4 * n_embd, n_embd)) * 0.02).astype(np.float32)
        tensors[f"{p}mlp.c_proj.bias"] = np.zeros(n_embd, dtype=np.float32)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _make_llama_model(tmp_path: Path) -> Path:
    rng = np.random.default_rng(123)
    hidden, heads, kv_heads, layers, inter, vocab = 8, 2, 1, 2, 16, 16
    head_dim = hidden // heads
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": hidden, "num_attention_heads": heads,
        "num_key_value_heads": kv_heads, "num_hidden_layers": layers,
        "intermediate_size": inter, "vocab_size": vocab,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    tensors: dict[str, Any] = {
        "model.embed_tokens.weight": rng.standard_normal((vocab, hidden)).astype(np.float32),
        "model.norm.weight": np.ones(hidden, dtype=np.float32),
        "lm_head.weight": rng.standard_normal((vocab, hidden)).astype(np.float32),
    }
    for i in range(layers):
        p = f"model.layers.{i}."
        tensors[f"{p}input_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        tensors[f"{p}self_attn.q_proj.weight"] = (rng.standard_normal((hidden, hidden)) * 0.02).astype(np.float32)
        tensors[f"{p}self_attn.k_proj.weight"] = (rng.standard_normal((kv_heads * head_dim, hidden)) * 0.02).astype(np.float32)
        tensors[f"{p}self_attn.v_proj.weight"] = (rng.standard_normal((kv_heads * head_dim, hidden)) * 0.02).astype(np.float32)
        tensors[f"{p}self_attn.o_proj.weight"] = (rng.standard_normal((hidden, hidden)) * 0.02).astype(np.float32)
        tensors[f"{p}post_attention_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        tensors[f"{p}mlp.gate_proj.weight"] = (rng.standard_normal((inter, hidden)) * 0.02).astype(np.float32)
        tensors[f"{p}mlp.up_proj.weight"] = (rng.standard_normal((inter, hidden)) * 0.02).astype(np.float32)
        tensors[f"{p}mlp.down_proj.weight"] = (rng.standard_normal((hidden, inter)) * 0.02).astype(np.float32)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


class TestGPT2EndToEnd:
    def test_full_pipeline(self, tmp_path: Path) -> None:
        model_dir = _make_gpt2_model(tmp_path)
        bundle = HuggingFaceLoader.load_from_dir(model_dir)
        executor = GPT2Executor(bundle.config)
        layer_ids = [l.layer_id for l in bundle.layers]

        with OffloadRuntime(
            layers=bundle.layers,
            backend=NullBackend(),
            storage=bundle.storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=executor,
        ) as runtime:
            # Embed
            token_ids = [0, 1, 2]
            hidden = executor.embed(
                token_ids,
                bundle.embed_weights["transformer.wte.weight"],
                bundle.embed_weights["transformer.wpe.weight"],
            )
            assert hidden.shape == (3, 8)

            # Run all layers
            hidden, metrics = runtime.run_inference(layer_ids, inputs=hidden)
            assert hidden.shape == (3, 8)
            assert metrics.layer_count == 2

            # LM head
            logits = executor.lm_head(
                hidden,
                bundle.head_weights["transformer.ln_f.weight"],
                bundle.head_weights["transformer.ln_f.bias"],
                bundle.head_weights["lm_head.weight"],
            )
            assert logits.shape == (3, 16)
            assert np.all(np.isfinite(logits))

            # Greedy next token
            next_token = int(np.argmax(logits[-1]))
            assert 0 <= next_token < 16

    def test_autoregressive_generation(self, tmp_path: Path) -> None:
        model_dir = _make_gpt2_model(tmp_path)
        bundle = HuggingFaceLoader.load_from_dir(model_dir)
        executor = GPT2Executor(bundle.config)
        layer_ids = [l.layer_id for l in bundle.layers]

        with OffloadRuntime(
            layers=bundle.layers,
            backend=NullBackend(),
            storage=bundle.storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=executor,
        ) as runtime:
            token_ids = [0, 1]
            for _ in range(3):
                hidden = executor.embed(
                    token_ids,
                    bundle.embed_weights["transformer.wte.weight"],
                    bundle.embed_weights["transformer.wpe.weight"],
                )
                hidden, _ = runtime.run_inference(layer_ids, inputs=hidden)
                logits = executor.lm_head(
                    hidden,
                    bundle.head_weights["transformer.ln_f.weight"],
                    bundle.head_weights["transformer.ln_f.bias"],
                    bundle.head_weights["lm_head.weight"],
                )
                next_token = int(np.argmax(logits[-1]))
                token_ids.append(next_token)

            assert len(token_ids) == 5
            assert all(0 <= t < 16 for t in token_ids)


class TestLlamaEndToEnd:
    def test_full_pipeline(self, tmp_path: Path) -> None:
        model_dir = _make_llama_model(tmp_path)
        bundle = HuggingFaceLoader.load_from_dir(model_dir)
        executor = LlamaExecutor(bundle.config)
        layer_ids = [l.layer_id for l in bundle.layers]

        with OffloadRuntime(
            layers=bundle.layers,
            backend=NullBackend(),
            storage=bundle.storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=executor,
        ) as runtime:
            token_ids = [0, 1, 2]
            hidden = executor.embed(
                token_ids,
                bundle.embed_weights["model.embed_tokens.weight"],
            )
            assert hidden.shape == (3, 8)

            hidden, metrics = runtime.run_inference(layer_ids, inputs=hidden)
            assert hidden.shape == (3, 8)
            assert metrics.layer_count == 2

            logits = executor.lm_head(
                hidden,
                bundle.head_weights["model.norm.weight"],
                bundle.head_weights["lm_head.weight"],
            )
            assert logits.shape == (3, 16)
            assert np.all(np.isfinite(logits))

    def test_autoregressive_generation(self, tmp_path: Path) -> None:
        model_dir = _make_llama_model(tmp_path)
        bundle = HuggingFaceLoader.load_from_dir(model_dir)
        executor = LlamaExecutor(bundle.config)
        layer_ids = [l.layer_id for l in bundle.layers]

        with OffloadRuntime(
            layers=bundle.layers,
            backend=NullBackend(),
            storage=bundle.storage,
            scheduler=LookaheadScheduler(lookahead=1),
            executor=executor,
        ) as runtime:
            token_ids = [0, 1]
            for _ in range(3):
                hidden = executor.embed(
                    token_ids,
                    bundle.embed_weights["model.embed_tokens.weight"],
                )
                hidden, _ = runtime.run_inference(layer_ids, inputs=hidden)
                logits = executor.lm_head(
                    hidden,
                    bundle.head_weights["model.norm.weight"],
                    bundle.head_weights["lm_head.weight"],
                )
                next_token = int(np.argmax(logits[-1]))
                token_ids.append(next_token)

            assert len(token_ids) == 5
            assert all(0 <= t < 16 for t in token_ids)
