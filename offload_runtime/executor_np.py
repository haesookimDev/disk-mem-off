"""Backward-compatible re-export shim.

All executor implementations have been moved to offload_runtime/executors/.
This module re-exports everything for backward compatibility.
"""
from __future__ import annotations

# Re-export shared math primitives
from offload_runtime.executors._common import (
    _readback_device,
    _unpack_tensors,
    gelu,
    layer_norm,
    linear,
    linear_t,
    np,
    repeat_kv,
    rms_norm,
    rope,
    silu,
    softmax,
)

# Re-export executor classes
from offload_runtime.executors.glm4 import GLM4Executor
from offload_runtime.executors.glm4 import LAYER_TENSORS as _GLM4_TENSOR_ORDER  # noqa: F401
from offload_runtime.executors.glm4_moe import GLM4MoeExecutor
from offload_runtime.executors.gpt2 import GPT2Executor
from offload_runtime.executors.gpt2 import LAYER_TENSORS as _GPT2_TENSOR_ORDER  # noqa: F401
from offload_runtime.executors.llama import LlamaExecutor
from offload_runtime.executors.llama import LAYER_TENSORS as _LLAMA_TENSOR_ORDER  # noqa: F401
from offload_runtime.executors.qwen3_next import Qwen3NextExecutor

__all__ = [
    # Helpers
    "_readback_device", "_unpack_tensors",
    "layer_norm", "rms_norm", "gelu", "silu", "softmax",
    "linear", "linear_t", "rope", "repeat_kv",
    # Executors
    "GPT2Executor", "LlamaExecutor", "GLM4Executor", "GLM4MoeExecutor",
    "Qwen3NextExecutor",
    # Tensor orders
    "_GPT2_TENSOR_ORDER", "_LLAMA_TENSOR_ORDER", "_GLM4_TENSOR_ORDER",
]
