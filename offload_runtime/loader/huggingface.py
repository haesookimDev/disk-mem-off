from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from offload_runtime.types import LayerSpec

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover
    safe_open = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from offload_runtime.loader.safetensors_storage import SafetensorsStorage


_DTYPE_ITEMSIZE: dict[str, int] = {
    "float16": 2, "float32": 4, "float64": 8,
    "bfloat16": 2, "int8": 1, "int16": 2, "int32": 4, "int64": 8,
}

_SUPPORTED_ARCHITECTURES: dict[str, str] = {
    "GPT2LMHeadModel": "gpt2",
    "LlamaForCausalLM": "llama",
    "Glm4ForCausalLM": "glm4",
}

# Canonical tensor order within a transformer block.
_GPT2_LAYER_TENSORS = [
    "ln_1.weight", "ln_1.bias",
    "attn.c_attn.weight", "attn.c_attn.bias",
    "attn.c_proj.weight", "attn.c_proj.bias",
    "ln_2.weight", "ln_2.bias",
    "mlp.c_fc.weight", "mlp.c_fc.bias",
    "mlp.c_proj.weight", "mlp.c_proj.bias",
]

_LLAMA_LAYER_TENSORS = [
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

_GLM4_LAYER_TENSORS = [
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


@dataclass(slots=True)
class ModelBundle:
    """Everything needed to run inference with OffloadRuntime."""

    architecture: str
    config: dict[str, Any]
    layers: list[LayerSpec]
    storage: Any  # SafetensorsStorage
    embed_weights: dict[str, Any]
    head_weights: dict[str, Any]
    tokenizer_path: Path | None = None


class HuggingFaceLoader:
    """Download and parse a HuggingFace model into OffloadRuntime components."""

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        cache_dir: str | None = None,
        compute_dtype: str = "float32",
    ) -> ModelBundle:
        if snapshot_download is None:
            raise RuntimeError("huggingface-hub is not installed")
        if safe_open is None:
            raise RuntimeError("safetensors is not installed")
        if np is None:
            raise RuntimeError("numpy is not installed")

        model_dir = Path(snapshot_download(
            model_id,
            cache_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.json", "model.safetensors.index.json"],
        ))

        config = cls._load_config(model_dir)
        architecture = cls._detect_architecture(config)

        st_files, tensor_to_filename = cls._find_safetensors(model_dir)
        tensor_to_file: dict[str, Path] = {
            name: model_dir / fname for name, fname in tensor_to_filename.items()
        }

        # Build tensor inventory (name → shape, dtype)
        tensor_info = cls._build_tensor_info(st_files)

        # Group into layers + non-layer tensors
        if architecture == "gpt2":
            layers, embed_names, head_names = cls._group_gpt2(config, tensor_info, compute_dtype)
        elif architecture == "llama":
            layers, embed_names, head_names = cls._group_llama(config, tensor_info, compute_dtype)
        elif architecture == "glm4":
            layers, embed_names, head_names = cls._group_glm4(config, tensor_info, compute_dtype)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Load non-layer weights eagerly
        embed_weights = cls._load_weights(embed_names, st_files, tensor_to_file, compute_dtype)
        head_weights = cls._load_weights(head_names, st_files, tensor_to_file, compute_dtype)

        # Handle weight tying
        if architecture == "gpt2" and "lm_head.weight" not in head_weights:
            head_weights["lm_head.weight"] = embed_weights["transformer.wte.weight"]
        elif architecture in ("llama", "glm4") and "lm_head.weight" not in head_weights:
            head_weights["lm_head.weight"] = embed_weights["model.embed_tokens.weight"]

        # Find tokenizer
        tokenizer_path: Path | None = None
        for candidate in ["tokenizer.json", "tokenizer.model"]:
            p = model_dir / candidate
            if p.exists():
                tokenizer_path = p
                break

        storage = SafetensorsStorage(
            safetensors_paths=st_files,
            layer_specs=layers,
            tensor_to_file=tensor_to_file,
            compute_dtype=compute_dtype,
        )

        return ModelBundle(
            architecture=architecture,
            config=config,
            layers=layers,
            storage=storage,
            embed_weights=embed_weights,
            head_weights=head_weights,
            tokenizer_path=tokenizer_path,
        )

    @classmethod
    def load_from_dir(
        cls,
        model_dir: str | Path,
        *,
        compute_dtype: str = "float32",
    ) -> ModelBundle:
        """Load from a local directory (no download). Useful for testing."""
        if safe_open is None:
            raise RuntimeError("safetensors is not installed")
        if np is None:
            raise RuntimeError("numpy is not installed")

        model_dir = Path(model_dir)
        config = cls._load_config(model_dir)
        architecture = cls._detect_architecture(config)

        st_files, tensor_to_filename = cls._find_safetensors(model_dir)
        tensor_to_file: dict[str, Path] = {
            name: model_dir / fname for name, fname in tensor_to_filename.items()
        }

        tensor_info = cls._build_tensor_info(st_files)

        if architecture == "gpt2":
            layers, embed_names, head_names = cls._group_gpt2(config, tensor_info, compute_dtype)
        elif architecture == "llama":
            layers, embed_names, head_names = cls._group_llama(config, tensor_info, compute_dtype)
        elif architecture == "glm4":
            layers, embed_names, head_names = cls._group_glm4(config, tensor_info, compute_dtype)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        embed_weights = cls._load_weights(embed_names, st_files, tensor_to_file, compute_dtype)
        head_weights = cls._load_weights(head_names, st_files, tensor_to_file, compute_dtype)

        if architecture == "gpt2" and "lm_head.weight" not in head_weights:
            head_weights["lm_head.weight"] = embed_weights["transformer.wte.weight"]
        elif architecture in ("llama", "glm4") and "lm_head.weight" not in head_weights:
            head_weights["lm_head.weight"] = embed_weights["model.embed_tokens.weight"]

        tokenizer_path: Path | None = None
        for candidate in ["tokenizer.json", "tokenizer.model"]:
            p = model_dir / candidate
            if p.exists():
                tokenizer_path = p
                break

        storage = SafetensorsStorage(
            safetensors_paths=st_files,
            layer_specs=layers,
            tensor_to_file=tensor_to_file,
            compute_dtype=compute_dtype,
        )

        return ModelBundle(
            architecture=architecture,
            config=config,
            layers=layers,
            storage=storage,
            embed_weights=embed_weights,
            head_weights=head_weights,
            tokenizer_path=tokenizer_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(model_dir: Path) -> dict[str, Any]:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_dir}")
        return json.loads(config_path.read_text(encoding="utf-8"))

    @staticmethod
    def _detect_architecture(config: dict[str, Any]) -> str:
        architectures = config.get("architectures", [])
        for arch in architectures:
            if arch in _SUPPORTED_ARCHITECTURES:
                return _SUPPORTED_ARCHITECTURES[arch]
        raise ValueError(
            f"Unsupported architecture(s): {architectures}. "
            f"Supported: {list(_SUPPORTED_ARCHITECTURES.keys())}"
        )

    @classmethod
    def _find_safetensors(cls, model_dir: Path) -> tuple[list[Path], dict[str, str]]:
        """Return (list of safetensors paths, {tensor_name: relative_filename})."""
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map: dict[str, str] = index["weight_map"]
            filenames = sorted(set(weight_map.values()))
            st_files = [model_dir / f for f in filenames]
            return st_files, weight_map

        single = model_dir / "model.safetensors"
        if single.exists():
            handle = safe_open(str(single), framework="numpy")
            tensor_names = list(handle.keys())
            tensor_map = {name: "model.safetensors" for name in tensor_names}
            return [single], tensor_map

        raise FileNotFoundError(
            f"No safetensors files found in {model_dir}. "
            "Expected model.safetensors or model.safetensors.index.json"
        )

    @classmethod
    def _build_tensor_info(
        cls, st_files: list[Path],
    ) -> dict[str, dict[str, Any]]:
        """Map tensor name → {shape, dtype} by inspecting all safetensors files."""
        info: dict[str, dict[str, Any]] = {}
        seen_files: set[Path] = set()
        for path in st_files:
            if path in seen_files:
                continue
            seen_files.add(path)
            handle = safe_open(str(path), framework="numpy")
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                info[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        return info

    @classmethod
    def _compute_nbytes(cls, shape: list[int], compute_dtype: str) -> int:
        n_elements = 1
        for d in shape:
            n_elements *= d
        return n_elements * _DTYPE_ITEMSIZE[compute_dtype]

    @classmethod
    def _group_gpt2(
        cls,
        config: dict[str, Any],
        tensor_info: dict[str, dict[str, Any]],
        compute_dtype: str,
    ) -> tuple[list[LayerSpec], list[str], list[str]]:
        n_layer = config["n_layer"]
        layers: list[LayerSpec] = []

        for i in range(n_layer):
            prefix = f"transformer.h.{i}."
            meta_list: list[dict[str, Any]] = []
            offset = 0

            for short_name in _GPT2_LAYER_TENSORS:
                full_name = prefix + short_name
                if full_name not in tensor_info:
                    continue
                shape = tensor_info[full_name]["shape"]
                nbytes = cls._compute_nbytes(shape, compute_dtype)
                meta_list.append({
                    "name": short_name,
                    "shape": shape,
                    "dtype": compute_dtype,
                    "offset": offset,
                    "nbytes": nbytes,
                })
                offset += nbytes

            layers.append(LayerSpec(
                layer_id=i,
                name=f"transformer.h.{i}",
                nbytes=offset,
                metadata={"tensors": meta_list, "full_prefix": prefix},
            ))

        embed_names = [n for n in tensor_info if n.startswith("transformer.w")]
        head_names = [n for n in tensor_info if n.startswith("transformer.ln_f") or n == "lm_head.weight"]

        return layers, embed_names, head_names

    @classmethod
    def _group_llama(
        cls,
        config: dict[str, Any],
        tensor_info: dict[str, dict[str, Any]],
        compute_dtype: str,
    ) -> tuple[list[LayerSpec], list[str], list[str]]:
        n_layer = config["num_hidden_layers"]
        layers: list[LayerSpec] = []

        for i in range(n_layer):
            prefix = f"model.layers.{i}."
            meta_list: list[dict[str, Any]] = []
            offset = 0

            for short_name in _LLAMA_LAYER_TENSORS:
                full_name = prefix + short_name
                if full_name not in tensor_info:
                    continue
                shape = tensor_info[full_name]["shape"]
                nbytes = cls._compute_nbytes(shape, compute_dtype)
                meta_list.append({
                    "name": short_name,
                    "shape": shape,
                    "dtype": compute_dtype,
                    "offset": offset,
                    "nbytes": nbytes,
                })
                offset += nbytes

            layers.append(LayerSpec(
                layer_id=i,
                name=f"model.layers.{i}",
                nbytes=offset,
                metadata={"tensors": meta_list, "full_prefix": prefix},
            ))

        embed_names = ["model.embed_tokens.weight"]
        head_names = [n for n in tensor_info if n == "model.norm.weight" or n == "lm_head.weight"]

        return layers, embed_names, head_names

    @classmethod
    def _group_glm4(
        cls,
        config: dict[str, Any],
        tensor_info: dict[str, dict[str, Any]],
        compute_dtype: str,
    ) -> tuple[list[LayerSpec], list[str], list[str]]:
        n_layer = config["num_hidden_layers"]
        layers: list[LayerSpec] = []

        for i in range(n_layer):
            prefix = f"model.layers.{i}."
            meta_list: list[dict[str, Any]] = []
            offset = 0

            for short_name in _GLM4_LAYER_TENSORS:
                full_name = prefix + short_name
                if full_name not in tensor_info:
                    continue
                shape = tensor_info[full_name]["shape"]
                nbytes = cls._compute_nbytes(shape, compute_dtype)
                meta_list.append({
                    "name": short_name,
                    "shape": shape,
                    "dtype": compute_dtype,
                    "offset": offset,
                    "nbytes": nbytes,
                })
                offset += nbytes

            layers.append(LayerSpec(
                layer_id=i,
                name=f"model.layers.{i}",
                nbytes=offset,
                metadata={"tensors": meta_list, "full_prefix": prefix},
            ))

        embed_names = ["model.embed_tokens.weight"]
        head_names = [n for n in tensor_info if n == "model.norm.weight" or n == "lm_head.weight"]

        return layers, embed_names, head_names

    @classmethod
    def _load_weights(
        cls,
        tensor_names: list[str],
        st_files: list[Path],
        tensor_to_file: dict[str, Path],
        compute_dtype: str,
    ) -> dict[str, Any]:
        """Load specific tensors eagerly as numpy arrays."""
        result: dict[str, Any] = {}
        target_dtype = np.dtype(compute_dtype)
        handles: dict[Path, Any] = {}

        for name in tensor_names:
            if name not in tensor_to_file:
                continue
            path = tensor_to_file[name]
            if path not in handles:
                handles[path] = safe_open(str(path), framework="numpy")
            arr = handles[path].get_tensor(name)
            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)
            result[name] = arr

        return result
