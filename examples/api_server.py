"""OpenAI / Anthropic compatible API server for the offload runtime.

Loads a HuggingFace model once at startup, then serves inference via
standard chat completion endpoints. Multiple users can connect concurrently;
requests are queued and processed sequentially (OffloadRuntime is not
thread-safe).

Usage:
    uv pip install fastapi "uvicorn[standard]"
    uv run python examples/api_server.py --model openai-community/gpt2 --port 8000
    uv run python examples/api_server.py --model zai-org/GLM-4-9B-0414 --backend auto
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: uv pip install numpy")

try:
    from tokenizers import Tokenizer
except ImportError:
    sys.exit("tokenizers is required: uv pip install tokenizers")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    sys.exit("fastapi is required: uv pip install fastapi 'uvicorn[standard]'")

from offload_runtime.backends import (
    CUDABackend,
    MPSBackend,
    NullBackend,
    ROCmBackend,
    detect_backend,
)
from offload_runtime.executor_np import (
    GLM4Executor,
    GLM4MoeExecutor,
    GPT2Executor,
    LlamaExecutor,
    Qwen3NextExecutor,
)
from offload_runtime.loader.huggingface import HuggingFaceLoader
from offload_runtime.runtime import OffloadRuntime
from offload_runtime.scheduler.lookahead import LookaheadScheduler

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_cli_args: argparse.Namespace | None = None

_BACKEND_MAP: dict[str, Any] = {
    "cuda": CUDABackend,
    "rocm": ROCmBackend,
    "mps": MPSBackend,
    "null": NullBackend,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI/Anthropic-compatible API server for offload runtime",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--local-dir", default=None, help="Load from local directory")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cuda", "rocm", "mps", "null"],
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--compute-dtype", default="float32")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pydantic models -- OpenAI
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage] = []
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False


class OpenAIChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage


# ---------------------------------------------------------------------------
# Pydantic models -- Anthropic
# ---------------------------------------------------------------------------


class AnthropicMessage(BaseModel):
    role: str
    content: str


class AnthropicRequest(BaseModel):
    model: str = ""
    messages: list[AnthropicMessage] = []
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False


class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class AnthropicResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: str
    usage: AnthropicUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_eos_token_ids(config: dict[str, Any]) -> set[int]:
    eos = config.get("eos_token_id")
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    if isinstance(eos, list):
        return set(eos)
    return set()


def _create_executor(bundle: Any) -> Any:
    executors: dict[str, type] = {
        "gpt2": GPT2Executor,
        "llama": LlamaExecutor,
        "glm4": GLM4Executor,
        "glm4_moe": GLM4MoeExecutor,
        "qwen3_next": Qwen3NextExecutor,
    }
    cls = executors.get(bundle.architecture)
    if cls is None:
        raise ValueError(f"Unsupported architecture: {bundle.architecture}")
    return cls(bundle.config)


def _create_backend(args: argparse.Namespace) -> Any:
    if args.backend == "auto":
        return detect_backend(device_id=args.device_id)
    cls = _BACKEND_MAP[args.backend]
    if args.backend in ("cuda", "rocm"):
        return cls(args.device_id)
    return cls()


def _format_chat_messages(messages: list[dict[str, str]], tokenizer_path: Any) -> str:
    """Convert chat messages to a prompt string.

    Tries to load a jinja2 chat_template from tokenizer_config.json first,
    then falls back to a simple role-prefixed format.
    """
    rendered = _try_chat_template(messages, tokenizer_path)
    if rendered is not None:
        return rendered

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _try_chat_template(
    messages: list[dict[str, str]], tokenizer_path: Any,
) -> str | None:
    if tokenizer_path is None:
        return None
    config_path = tokenizer_path.parent / "tokenizer_config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        template_str = config.get("chat_template")
        if not template_str:
            return None
        from jinja2 import BaseLoader, Environment

        env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(
            Exception(msg)
        )
        tmpl = env.from_string(template_str)
        return tmpl.render(
            messages=messages,
            add_generation_prompt=True,
            bos_token="",
            eos_token="",
        )
    except Exception:
        return None


def _sample(logits: Any, temperature: float, top_p: float) -> int:
    if temperature < 1e-7:
        return int(np.argmax(logits))

    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()

    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cum = np.cumsum(sorted_probs)
        cutoff = int(np.searchsorted(cum, top_p)) + 1
        mask = np.zeros_like(probs, dtype=bool)
        mask[sorted_idx[:cutoff]] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()

    return int(np.random.choice(len(probs), p=probs))


def _forward_step(
    bundle: Any,
    runtime: OffloadRuntime,
    executor: Any,
    layer_ids: list[int],
    token_ids: list[int],
    temperature: float,
    top_p: float,
) -> int:
    """One autoregressive step: embed -> layers -> head -> sample."""
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

    hidden, _ = runtime.run_inference(layer_ids, inputs=hidden)

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

    return _sample(logits[-1], temperature, top_p)


# ---------------------------------------------------------------------------
# Inference queue & worker
# ---------------------------------------------------------------------------


@dataclass
class InferenceRequest:
    token_ids: list[int]
    max_tokens: int
    temperature: float
    top_p: float
    stop_token_ids: set[int]
    output_queue: asyncio.Queue[int | None | Exception] = field(
        default_factory=asyncio.Queue,
    )


async def _generate_tokens(
    bundle: Any,
    runtime: OffloadRuntime,
    executor: Any,
    layer_ids: list[int],
    req: InferenceRequest,
) -> None:
    token_ids = list(req.token_ids)
    for _ in range(req.max_tokens):
        next_token = await asyncio.to_thread(
            _forward_step,
            bundle,
            runtime,
            executor,
            layer_ids,
            token_ids,
            req.temperature,
            req.top_p,
        )
        token_ids.append(next_token)
        await req.output_queue.put(next_token)
        if next_token in req.stop_token_ids:
            break
    await req.output_queue.put(None)


async def _inference_worker(
    bundle: Any,
    runtime: OffloadRuntime,
    executor: Any,
    layer_ids: list[int],
    queue: asyncio.Queue[InferenceRequest],
) -> None:
    while True:
        req = await queue.get()
        try:
            await _generate_tokens(bundle, runtime, executor, layer_ids, req)
        except Exception as exc:
            await req.output_queue.put(exc)
        finally:
            queue.task_done()


# ---------------------------------------------------------------------------
# SSE streaming generators
# ---------------------------------------------------------------------------


async def _openai_stream(
    output_queue: asyncio.Queue[int | None | Exception],
    request_id: str,
    created: int,
    model: str,
    tokenizer: Any,
    eos_ids: set[int],
) -> AsyncGenerator[str, None]:
    def _chunk(delta: dict[str, Any], finish: str | None) -> str:
        obj = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish},
            ],
        }
        return f"data: {json.dumps(obj)}\n\n"

    yield _chunk({"role": "assistant", "content": ""}, None)

    while True:
        item = await output_queue.get()
        if item is None:
            yield _chunk({}, "length")
            break
        if isinstance(item, Exception):
            yield _chunk({}, "stop")
            break
        if item in eos_ids:
            yield _chunk({}, "stop")
            break
        text = tokenizer.decode([item])
        yield _chunk({"content": text}, None)

    yield "data: [DONE]\n\n"


async def _anthropic_stream(
    output_queue: asyncio.Queue[int | None | Exception],
    request_id: str,
    model: str,
    input_tokens: int,
    tokenizer: Any,
    eos_ids: set[int],
) -> AsyncGenerator[str, None]:
    def _evt(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    output_count = 0

    yield _evt(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": request_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )
    yield _evt(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    )

    stop_reason = "max_tokens"
    while True:
        item = await output_queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            break
        output_count += 1
        if item in eos_ids:
            stop_reason = "end_turn"
            break
        text = tokenizer.decode([item])
        yield _evt(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        )

    yield _evt("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _evt(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": output_count},
        },
    )
    yield _evt("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Shared request handling
# ---------------------------------------------------------------------------


async def _collect_tokens(
    output_queue: asyncio.Queue[int | None | Exception],
    eos_ids: set[int],
) -> tuple[list[int], str]:
    """Drain the output queue; return (generated_ids, finish_reason)."""
    generated: list[int] = []
    finish = "length"
    while True:
        item = await output_queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise HTTPException(status_code=500, detail=str(item))
        generated.append(item)
        if item in eos_ids:
            finish = "stop"
            break
    return generated, finish


async def _submit_request(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[InferenceRequest, int]:
    """Tokenize messages and submit an inference request to the queue."""
    prompt_text = _format_chat_messages(messages, _bundle.tokenizer_path)
    token_ids = _tokenizer.encode(prompt_text).ids
    prompt_count = len(token_ids)

    req = InferenceRequest(
        token_ids=list(token_ids),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=_eos_token_ids,
    )
    await _inference_queue.put(req)
    return req, prompt_count


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

# Module-level state (populated during lifespan)
_bundle: Any = None
_runtime: OffloadRuntime | None = None
_executor: Any = None
_tokenizer: Any = None
_layer_ids: list[int] = []
_eos_token_ids: set[int] = set()
_inference_queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
_model_id: str = ""


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    global _bundle, _runtime, _executor, _tokenizer
    global _layer_ids, _eos_token_ids, _inference_queue, _model_id

    args = _cli_args or _parse_args()
    _model_id = args.model

    print(f"Loading model: {args.model}")
    t0 = time.perf_counter()
    if args.local_dir:
        bundle = HuggingFaceLoader.load_from_dir(
            args.local_dir, compute_dtype=args.compute_dtype,
        )
    else:
        bundle = HuggingFaceLoader.load(
            args.model, compute_dtype=args.compute_dtype,
        )
    print(f"  Architecture: {bundle.architecture}")
    print(f"  Layers: {len(bundle.layers)}")
    print(f"  Load time: {time.perf_counter() - t0:.2f}s")

    executor = _create_executor(bundle)
    backend = _create_backend(args)
    print(f"  Backend: {backend.name}")

    if bundle.tokenizer_path is not None:
        tokenizer = Tokenizer.from_file(str(bundle.tokenizer_path))
    else:
        sys.exit("Tokenizer not found in model directory")

    layer_ids = [layer.layer_id for layer in bundle.layers]
    runtime = OffloadRuntime(
        layers=bundle.layers,
        backend=backend,
        storage=bundle.storage,
        scheduler=LookaheadScheduler(lookahead=2),
        executor=executor,
    )

    _bundle = bundle
    _runtime = runtime
    _executor = executor
    _tokenizer = tokenizer
    _layer_ids = layer_ids
    _eos_token_ids = _get_eos_token_ids(bundle.config)
    _inference_queue = asyncio.Queue()

    worker = asyncio.create_task(
        _inference_worker(bundle, runtime, executor, layer_ids, _inference_queue),
    )
    print(f"\nServer ready. Model: {args.model}")
    print(f"  POST /v1/chat/completions  (OpenAI)")
    print(f"  POST /v1/messages          (Anthropic)")
    print(f"  GET  /v1/models\n")

    yield

    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass
    runtime.close()


app = FastAPI(title="disk-mem-off API", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": _model_id,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(body: OpenAIChatRequest):
    messages = [m.model_dump() for m in body.messages]
    req, prompt_count = await _submit_request(
        messages, body.max_tokens, body.temperature, body.top_p,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = body.model or _model_id

    if body.stream:
        return StreamingResponse(
            _openai_stream(
                req.output_queue, request_id, created, model,
                _tokenizer, _eos_token_ids,
            ),
            media_type="text/event-stream",
        )

    generated, finish = await _collect_tokens(req.output_queue, _eos_token_ids)
    decode_ids = generated[:-1] if finish == "stop" and generated else generated
    output_text = _tokenizer.decode(decode_ids)

    return OpenAIChatResponse(
        id=request_id,
        created=created,
        model=model,
        choices=[
            OpenAIChoice(
                index=0,
                message=ChatMessage(role="assistant", content=output_text),
                finish_reason=finish,
            )
        ],
        usage=OpenAIUsage(
            prompt_tokens=prompt_count,
            completion_tokens=len(generated),
            total_tokens=prompt_count + len(generated),
        ),
    )


@app.post("/v1/messages")
async def anthropic_messages(body: AnthropicRequest):
    messages = [m.model_dump() for m in body.messages]
    req, prompt_count = await _submit_request(
        messages, body.max_tokens, body.temperature, body.top_p,
    )

    request_id = f"msg_{uuid.uuid4().hex[:12]}"
    model = body.model or _model_id

    if body.stream:
        return StreamingResponse(
            _anthropic_stream(
                req.output_queue, request_id, model, prompt_count,
                _tokenizer, _eos_token_ids,
            ),
            media_type="text/event-stream",
        )

    generated, finish = await _collect_tokens(req.output_queue, _eos_token_ids)
    stop_reason = "end_turn" if finish == "stop" else "max_tokens"
    decode_ids = generated[:-1] if finish == "stop" and generated else generated
    output_text = _tokenizer.decode(decode_ids)

    return AnthropicResponse(
        id=request_id,
        content=[AnthropicContentBlock(text=output_text)],
        model=model,
        stop_reason=stop_reason,
        usage=AnthropicUsage(
            input_tokens=prompt_count,
            output_tokens=len(generated),
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global _cli_args
    _cli_args = _parse_args()

    import uvicorn

    uvicorn.run(app, host=_cli_args.host, port=_cli_args.port)


if __name__ == "__main__":
    main()
