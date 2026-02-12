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
    from offload_runtime.loader.huggingface import HuggingFaceLoader, ModelBundle


# ---------------------------------------------------------------------------
# Helpers for creating fake model directories
# ---------------------------------------------------------------------------

def _make_gpt2_model(tmp_path: Path, n_embd: int = 8, n_head: int = 2, n_layer: int = 2) -> Path:
    """Create a minimal GPT-2 model directory with safetensors and config."""
    rng = np.random.default_rng(42)
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "n_positions": 16,
        "vocab_size": 32,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    tensors: dict[str, Any] = {}
    # Embedding
    tensors["transformer.wte.weight"] = rng.standard_normal((32, n_embd)).astype(np.float32)
    tensors["transformer.wpe.weight"] = rng.standard_normal((16, n_embd)).astype(np.float32)
    # Final norm
    tensors["transformer.ln_f.weight"] = np.ones(n_embd, dtype=np.float32)
    tensors["transformer.ln_f.bias"] = np.zeros(n_embd, dtype=np.float32)

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


def _make_llama_model(
    tmp_path: Path,
    hidden_size: int = 8,
    num_heads: int = 2,
    num_kv_heads: int = 1,
    num_layers: int = 2,
    intermediate_size: int = 16,
) -> Path:
    """Create a minimal LLaMA model directory."""
    rng = np.random.default_rng(123)
    head_dim = hidden_size // num_heads
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate_size,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    tensors: dict[str, Any] = {}
    tensors["model.embed_tokens.weight"] = rng.standard_normal((32, hidden_size)).astype(np.float32)
    tensors["model.norm.weight"] = np.ones(hidden_size, dtype=np.float32)
    tensors["lm_head.weight"] = rng.standard_normal((32, hidden_size)).astype(np.float32)

    for i in range(num_layers):
        p = f"model.layers.{i}."
        tensors[f"{p}input_layernorm.weight"] = np.ones(hidden_size, dtype=np.float32)
        tensors[f"{p}self_attn.q_proj.weight"] = (rng.standard_normal((hidden_size, hidden_size)) * 0.02).astype(np.float32)
        tensors[f"{p}self_attn.k_proj.weight"] = (rng.standard_normal((num_kv_heads * head_dim, hidden_size)) * 0.02).astype(np.float32)
        tensors[f"{p}self_attn.v_proj.weight"] = (rng.standard_normal((num_kv_heads * head_dim, hidden_size)) * 0.02).astype(np.float32)
        tensors[f"{p}self_attn.o_proj.weight"] = (rng.standard_normal((hidden_size, hidden_size)) * 0.02).astype(np.float32)
        tensors[f"{p}post_attention_layernorm.weight"] = np.ones(hidden_size, dtype=np.float32)
        tensors[f"{p}mlp.gate_proj.weight"] = (rng.standard_normal((intermediate_size, hidden_size)) * 0.02).astype(np.float32)
        tensors[f"{p}mlp.up_proj.weight"] = (rng.standard_normal((intermediate_size, hidden_size)) * 0.02).astype(np.float32)
        tensors[f"{p}mlp.down_proj.weight"] = (rng.standard_normal((hidden_size, intermediate_size)) * 0.02).astype(np.float32)

    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHuggingFaceLoaderGPT2:
    def test_detect_gpt2_architecture(self, tmp_path: Path) -> None:
        config = {"architectures": ["GPT2LMHeadModel"]}
        arch = HuggingFaceLoader._detect_architecture(config)
        assert arch == "gpt2"

    def test_load_gpt2_from_dir(self, tmp_path: Path) -> None:
        model_dir = _make_gpt2_model(tmp_path, n_embd=8, n_head=2, n_layer=2)
        bundle = HuggingFaceLoader.load_from_dir(model_dir)

        assert bundle.architecture == "gpt2"
        assert len(bundle.layers) == 2
        assert bundle.config["n_embd"] == 8

        # Verify LayerSpec structure
        for i, layer in enumerate(bundle.layers):
            assert layer.layer_id == i
            assert layer.name == f"transformer.h.{i}"
            assert layer.nbytes > 0
            tensor_names = [m["name"] for m in layer.metadata["tensors"]]
            assert "ln_1.weight" in tensor_names
            assert "attn.c_attn.weight" in tensor_names
            assert "mlp.c_proj.weight" in tensor_names

        # Verify embed weights
        assert "transformer.wte.weight" in bundle.embed_weights
        assert "transformer.wpe.weight" in bundle.embed_weights
        assert bundle.embed_weights["transformer.wte.weight"].shape == (32, 8)

        # Verify head weights (lm_head tied to wte)
        assert "lm_head.weight" in bundle.head_weights
        assert "transformer.ln_f.weight" in bundle.head_weights

        # Verify storage works
        buf = bundle.storage.get(0)
        assert buf.nbytes == bundle.layers[0].nbytes
        bundle.storage.close()

    def test_gpt2_layer_tensor_count(self, tmp_path: Path) -> None:
        model_dir = _make_gpt2_model(tmp_path, n_embd=8, n_head=2, n_layer=1)
        bundle = HuggingFaceLoader.load_from_dir(model_dir)
        # GPT-2 has 12 tensors per layer
        assert len(bundle.layers[0].metadata["tensors"]) == 12
        bundle.storage.close()

    def test_gpt2_weight_tying(self, tmp_path: Path) -> None:
        """lm_head.weight should be tied to wte.weight when not present."""
        model_dir = _make_gpt2_model(tmp_path, n_embd=8, n_head=2, n_layer=1)
        bundle = HuggingFaceLoader.load_from_dir(model_dir)
        np.testing.assert_array_equal(
            bundle.head_weights["lm_head.weight"],
            bundle.embed_weights["transformer.wte.weight"],
        )
        bundle.storage.close()


class TestHuggingFaceLoaderLlama:
    def test_detect_llama_architecture(self, tmp_path: Path) -> None:
        config = {"architectures": ["LlamaForCausalLM"]}
        arch = HuggingFaceLoader._detect_architecture(config)
        assert arch == "llama"

    def test_load_llama_from_dir(self, tmp_path: Path) -> None:
        model_dir = _make_llama_model(tmp_path)
        bundle = HuggingFaceLoader.load_from_dir(model_dir)

        assert bundle.architecture == "llama"
        assert len(bundle.layers) == 2
        assert bundle.config["hidden_size"] == 8

        for i, layer in enumerate(bundle.layers):
            assert layer.layer_id == i
            tensor_names = [m["name"] for m in layer.metadata["tensors"]]
            assert "input_layernorm.weight" in tensor_names
            assert "self_attn.q_proj.weight" in tensor_names
            assert "mlp.gate_proj.weight" in tensor_names

        assert "model.embed_tokens.weight" in bundle.embed_weights
        assert "lm_head.weight" in bundle.head_weights
        assert "model.norm.weight" in bundle.head_weights

        buf = bundle.storage.get(0)
        assert buf.nbytes == bundle.layers[0].nbytes
        bundle.storage.close()

    def test_llama_layer_tensor_count(self, tmp_path: Path) -> None:
        model_dir = _make_llama_model(tmp_path, num_layers=1)
        bundle = HuggingFaceLoader.load_from_dir(model_dir)
        # LLaMA has 9 tensors per layer
        assert len(bundle.layers[0].metadata["tensors"]) == 9
        bundle.storage.close()


class TestHuggingFaceLoaderErrors:
    def test_unsupported_architecture(self, tmp_path: Path) -> None:
        config = {"architectures": ["SomeUnknownModel"]}
        (tmp_path / "config.json").write_text(json.dumps(config))
        save_file({"x": np.array([1.0], dtype=np.float32)}, str(tmp_path / "model.safetensors"))
        with pytest.raises(ValueError, match="Unsupported"):
            HuggingFaceLoader.load_from_dir(tmp_path)

    def test_missing_config(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="config.json"):
            HuggingFaceLoader.load_from_dir(tmp_path)

    def test_missing_safetensors(self, tmp_path: Path) -> None:
        config = {"architectures": ["GPT2LMHeadModel"], "n_layer": 1}
        (tmp_path / "config.json").write_text(json.dumps(config))
        with pytest.raises(FileNotFoundError, match="safetensors"):
            HuggingFaceLoader.load_from_dir(tmp_path)

    def test_sharded_index_parsing(self, tmp_path: Path) -> None:
        """Verify that sharded index.json is parsed correctly."""
        rng = np.random.default_rng(42)
        config = {"architectures": ["GPT2LMHeadModel"], "n_embd": 4, "n_head": 1, "n_layer": 1, "n_positions": 4, "vocab_size": 8}
        (tmp_path / "config.json").write_text(json.dumps(config))

        # Create two shards
        shard1_tensors = {
            "transformer.wte.weight": rng.standard_normal((8, 4)).astype(np.float32),
            "transformer.wpe.weight": rng.standard_normal((4, 4)).astype(np.float32),
            "transformer.h.0.ln_1.weight": np.ones(4, dtype=np.float32),
            "transformer.h.0.ln_1.bias": np.zeros(4, dtype=np.float32),
            "transformer.h.0.attn.c_attn.weight": (rng.standard_normal((4, 12)) * 0.02).astype(np.float32),
            "transformer.h.0.attn.c_attn.bias": np.zeros(12, dtype=np.float32),
            "transformer.h.0.attn.c_proj.weight": (rng.standard_normal((4, 4)) * 0.02).astype(np.float32),
            "transformer.h.0.attn.c_proj.bias": np.zeros(4, dtype=np.float32),
        }
        shard2_tensors = {
            "transformer.h.0.ln_2.weight": np.ones(4, dtype=np.float32),
            "transformer.h.0.ln_2.bias": np.zeros(4, dtype=np.float32),
            "transformer.h.0.mlp.c_fc.weight": (rng.standard_normal((4, 16)) * 0.02).astype(np.float32),
            "transformer.h.0.mlp.c_fc.bias": np.zeros(16, dtype=np.float32),
            "transformer.h.0.mlp.c_proj.weight": (rng.standard_normal((16, 4)) * 0.02).astype(np.float32),
            "transformer.h.0.mlp.c_proj.bias": np.zeros(4, dtype=np.float32),
            "transformer.ln_f.weight": np.ones(4, dtype=np.float32),
            "transformer.ln_f.bias": np.zeros(4, dtype=np.float32),
        }
        save_file(shard1_tensors, str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file(shard2_tensors, str(tmp_path / "model-00002-of-00002.safetensors"))

        # Build weight_map
        weight_map: dict[str, str] = {}
        for name in shard1_tensors:
            weight_map[name] = "model-00001-of-00002.safetensors"
        for name in shard2_tensors:
            weight_map[name] = "model-00002-of-00002.safetensors"

        index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        bundle = HuggingFaceLoader.load_from_dir(tmp_path)
        assert len(bundle.layers) == 1
        assert len(bundle.layers[0].metadata["tensors"]) == 12

        # Storage should read from both shards
        buf = bundle.storage.get(0)
        assert buf.nbytes == bundle.layers[0].nbytes
        bundle.storage.close()
