"""Demo: Text generation using the offload runtime.

Downloads a HuggingFace model, streams weights layer-by-layer through
the best available backend, and performs autoregressive generation using NumPy.

Usage:
    uv run python examples/demo_inference.py --model openai-community/gpt2 --prompt "Hello, I am" --max-tokens 50
    uv run python examples/demo_inference.py --model meta-llama/Llama-3.2-1B --prompt "Once upon a" --max-tokens 30
    uv run python examples/demo_inference.py --model zai-org/GLM-4.7 --backend cuda --device-id 0
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: uv pip install numpy")

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

from offload_runtime.backends import (
    CUDABackend,
    MPSBackend,
    NullBackend,
    ROCmBackend,
    detect_backend,
)
from offload_runtime.executor_np import (
    GLM4Executor, GLM4MoeExecutor, GPT2Executor, LlamaExecutor, Qwen3NextExecutor,
)
from offload_runtime.loader.huggingface import HuggingFaceLoader
from offload_runtime.runtime import OffloadRuntime
from offload_runtime.scheduler.lookahead import LookaheadScheduler

_BACKEND_MAP: dict[str, Any] = {
    "cuda": CUDABackend,
    "rocm": ROCmBackend,
    "mps": MPSBackend,
    "null": NullBackend,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offload-runtime text generation demo")
    parser.add_argument("--model", required=True, help="HuggingFace model ID (e.g. openai-community/gpt2)")
    parser.add_argument("--prompt", default="Hello, I am", help="Input prompt text")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--compute-dtype", default="float32", help="Compute dtype")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "rocm", "mps", "null"],
                        help="Device backend (default: auto-detect)")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID (for cuda/rocm)")
    parser.add_argument("--local-dir", default=None, help="Load from local directory instead of downloading")
    args = parser.parse_args()

    # --- Load model ---
    print(f"Loading model: {args.model}")
    t0 = time.perf_counter()
    if args.local_dir:
        bundle = HuggingFaceLoader.load_from_dir(args.local_dir, compute_dtype=args.compute_dtype)
    else:
        bundle = HuggingFaceLoader.load(args.model, compute_dtype=args.compute_dtype)
    load_time = time.perf_counter() - t0
    print(f"  Architecture: {bundle.architecture}")
    print(f"  Layers: {len(bundle.layers)}")
    print(f"  Load time: {load_time:.2f}s")

    # --- Select executor ---
    if bundle.architecture == "gpt2":
        executor: Any = GPT2Executor(bundle.config)
    elif bundle.architecture == "llama":
        executor = LlamaExecutor(bundle.config)
    elif bundle.architecture == "glm4":
        executor = GLM4Executor(bundle.config)
    elif bundle.architecture == "glm4_moe":
        executor = GLM4MoeExecutor(bundle.config)
    elif bundle.architecture == "qwen3_next":
        executor = Qwen3NextExecutor(bundle.config)
    else:
        sys.exit(f"Unsupported architecture: {bundle.architecture}")

    # --- Tokenize ---
    if Tokenizer is not None and bundle.tokenizer_path is not None:
        tokenizer = Tokenizer.from_file(str(bundle.tokenizer_path))
        token_ids: list[int] = tokenizer.encode(args.prompt).ids
        print(f"  Tokenizer: loaded from {bundle.tokenizer_path.name}")
    else:
        # Fallback: ASCII byte encoding
        print("  Tokenizer: not available, using ASCII byte encoding")
        tokenizer = None
        vocab_size = bundle.config.get("vocab_size", bundle.config.get("vocab_size", 50257))
        token_ids = [min(ord(c), vocab_size - 1) for c in args.prompt]

    print(f"  Input tokens: {len(token_ids)}")
    print()

    # --- Select backend ---
    if args.backend == "auto":
        backend = detect_backend(device_id=args.device_id)
    else:
        cls = _BACKEND_MAP[args.backend]
        if cls is None:
            sys.exit(f"Backend '{args.backend}' is not available (missing dependency)")
        backend = cls() if args.backend == "mps" else cls(args.device_id) if args.backend in ("cuda", "rocm") else cls()
    print(f"  Backend: {backend.name}")
    print()

    # --- Create runtime ---
    layer_ids = [layer.layer_id for layer in bundle.layers]

    with OffloadRuntime(
        layers=bundle.layers,
        backend=backend,
        storage=bundle.storage,
        scheduler=LookaheadScheduler(lookahead=2),
        executor=executor,
    ) as runtime:
        # --- Generation loop ---
        print("Generating...")
        gen_start = time.perf_counter()
        total_metrics = None

        for step in range(args.max_tokens):
            # Embed
            if bundle.architecture == "gpt2":
                hidden = executor.embed(
                    token_ids,
                    bundle.embed_weights["transformer.wte.weight"],
                    bundle.embed_weights["transformer.wpe.weight"],
                )
            else:
                hidden = executor.embed(
                    token_ids,
                    bundle.embed_weights["model.embed_tokens.weight"],
                )

            # Forward all layers
            hidden, metrics = runtime.run_inference(layer_ids, inputs=hidden)

            # LM head
            if bundle.architecture == "gpt2":
                logits = executor.lm_head(
                    hidden,
                    bundle.head_weights["transformer.ln_f.weight"],
                    bundle.head_weights["transformer.ln_f.bias"],
                    bundle.head_weights["lm_head.weight"],
                )
            else:
                logits = executor.lm_head(
                    hidden,
                    bundle.head_weights["model.norm.weight"],
                    bundle.head_weights["lm_head.weight"],
                )

            # Greedy decode
            next_token = int(np.argmax(logits[-1]))
            token_ids.append(next_token)

            # Print progress
            if (step + 1) % 10 == 0:
                elapsed = time.perf_counter() - gen_start
                print(f"  Step {step + 1}/{args.max_tokens} ({elapsed:.1f}s)")

        gen_time = time.perf_counter() - gen_start

    # --- Decode and print result ---
    print()
    print("=" * 60)
    if tokenizer is not None:
        output_text = tokenizer.decode(token_ids)
        print(f"Output: {output_text}")
    else:
        print(f"Output token IDs: {token_ids}")
    print("=" * 60)
    print(f"Generation time: {gen_time:.2f}s ({args.max_tokens / gen_time:.1f} tokens/s)")
    print(f"Last step metrics: {metrics.layer_count} layers, "
          f"{metrics.transferred_bytes / 1e6:.1f} MB transferred")


if __name__ == "__main__":
    main()
