# disk-mem-off

Memory offloading runtime for large model inference with limited VRAM.

VRAM이 부족한 환경에서도 대형 언어 모델을 실행할 수 있도록 레이어 단위로 가중치를 스트리밍하는 런타임입니다. 호스트 메모리(RAM) 또는 외장 SSD에 가중치를 두고, 필요한 레이어만 GPU로 전송하여 추론을 수행합니다.

## Features

- **Layer-wise weight streaming** — 전체 모델을 VRAM에 올리지 않고 레이어 단위로 전송/실행/해제
- **Lookahead prefetching** — 다음 레이어를 미리 전송하여 전송/연산 오버랩으로 지연 최소화
- **Multi-backend support** — CUDA, ROCm, MPS(Apple Silicon), NullBackend(CPU) 자동 감지
- **HuggingFace integration** — 모델 ID만으로 자동 다운로드 및 safetensors 로딩
- **External storage support** — 외장 SSD 등 로컬 디렉토리에서 직접 모델 로드 가능

### Supported Models

| Model | Architecture | Key Features |
|-------|-------------|--------------|
| [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | GPT2LMHeadModel | MHA + MLP |
| [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | LlamaForCausalLM | GQA + RoPE + SwiGLU |
| [THUDM/glm-4-9b](https://huggingface.co/THUDM/glm-4-9b) | Glm4ForCausalLM | GQA + RoPE + SwiGLU |
| [zai-org/GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) | Glm4MoeForCausalLM | GQA + QK norm + 160-expert MoE |
| [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | Qwen3NextForCausalLM | Gated DeltaNet + Gated GQA + 512-expert MoE |

## Requirements

- Python >= 3.13
- NumPy, safetensors, huggingface-hub, tokenizers

## Installation

```bash
# uv (recommended)
uv sync

# pip
pip install -e .

# GPU backend dependencies (optional)
pip install -e ".[cuda]"    # NVIDIA
pip install -e ".[rocm]"    # AMD
pip install -e ".[mps]"     # Apple Silicon
```

## Quick Start

### Dry run (no model download)

```bash
uv run python -m examples.dry_run
```

### Text generation

```bash
# GPT-2 (small, ~500MB)
uv run python examples/demo_inference.py \
  --model openai-community/gpt2 \
  --prompt "Hello, I am" \
  --max-tokens 50

# LLaMA 3.2 1B
uv run python examples/demo_inference.py \
  --model meta-llama/Llama-3.2-1B \
  --prompt "Once upon a time" \
  --max-tokens 30

# GLM-4.7 MoE (280B params, ~159GB)
uv run python examples/demo_inference.py \
  --model zai-org/GLM-4.7 \
  --prompt "Hello" \
  --max-tokens 20

# Qwen3-Coder-Next (hybrid Gated DeltaNet + MoE, ~159GB)
uv run python examples/demo_inference.py \
  --model Qwen/Qwen3-Coder-Next \
  --prompt "def fibonacci(n):" \
  --max-tokens 30
```

### Backend selection

```bash
# Auto-detect (default): CUDA → ROCm → MPS → NullBackend
uv run python examples/demo_inference.py --model openai-community/gpt2

# Specify backend
uv run python examples/demo_inference.py --model openai-community/gpt2 --backend cuda --device-id 0
uv run python examples/demo_inference.py --model openai-community/gpt2 --backend mps
uv run python examples/demo_inference.py --model openai-community/gpt2 --backend null
```

### Loading from local directory / external SSD

대형 모델을 외장 SSD에 저장해두고 사용할 수 있습니다.

```bash
# 1. Download model to external storage
huggingface-cli download zai-org/GLM-4.7 --local-dir /Volumes/MySSD/models/GLM-4.7

# 2. Run from local directory
uv run python examples/demo_inference.py \
  --model zai-org/GLM-4.7 \
  --local-dir /Volumes/MySSD/models/GLM-4.7 \
  --prompt "Hello" \
  --max-tokens 20
```

### CLI options

```
--model          HuggingFace model ID (required)
--prompt         Input prompt text (default: "Hello, I am")
--max-tokens     Maximum tokens to generate (default: 50)
--compute-dtype  Compute dtype: float32, float16 (default: float32)
--backend        Device backend: auto, cuda, rocm, mps, null (default: auto)
--device-id      GPU device ID for multi-GPU (default: 0)
--local-dir      Load from local directory instead of downloading
```

## Architecture

```
Host Memory (RAM / NVMe SSD)
    │
    ▼
Storage (SafetensorsStorage / ShardedMMapStorage)
    │
    ▼
Scheduler (Lookahead / Block / CostAware)
    │  prefetch next W layers
    ▼
Pinned Host Buffer Pool
    │  async H2D copy (transfer stream)
    ▼
Device Buffer Pool (GPU VRAM)
    │  compute (compute stream)
    ▼
Layer Executor (GPT2 / LLaMA / GLM4 / GLM4MoE / Qwen3Next)
    │
    ▼
Output activations → next layer or LM head
```

### Core Components

| Component | Description |
|-----------|-------------|
| **Backend** | GPU 추상화 (CUDA/ROCm/MPS/Null). 메모리 할당, 비동기 전송, 스트림/이벤트 관리 |
| **Storage** | 가중치 저장소. SafeTensors 파일에서 레이어별 텐서를 로드 |
| **Scheduler** | 프리페치 전략. Lookahead, Block, CostAware, Reverse(역전파용) |
| **Executor** | 모델별 NumPy 기반 순전파 구현. 레이어 단위로 실행 |
| **Runtime** | 코어 오프로딩 루프. 스토리지→전송→실행→해제 파이프라인 관리 |
| **BufferPool** | GPU 메모리 재사용 풀. 반복적인 할당/해제 오버헤드 제거 |
| **Quantize** | INT8/FP16 역양자화. PCIe 전송량 감소 |

## Project Structure

```
disk-mem-off/
├── offload_runtime/
│   ├── __init__.py              # Public API exports
│   ├── runtime.py               # Core offload loop
│   ├── types.py                 # LayerSpec, DeviceBuffer, HostBuffer, LoRASpec
│   ├── training.py              # LoRA training offload
│   ├── backends/                # GPU backend adapters
│   │   ├── base.py              #   Abstract DeviceBackend
│   │   ├── cuda_backend.py      #   NVIDIA CUDA
│   │   ├── rocm_backend.py      #   AMD ROCm (HIP)
│   │   ├── mps_backend.py       #   Apple Metal
│   │   └── null_backend.py      #   CPU fallback (testing)
│   ├── storage/                 # Weight storage
│   │   ├── in_memory.py         #   In-memory (testing)
│   │   └── sharded_mmap.py      #   Memory-mapped shards
│   ├── scheduler/               # Prefetch scheduling
│   │   ├── lookahead.py         #   Fixed lookahead
│   │   ├── block_scheduler.py   #   Block-wise grouping
│   │   ├── cost_aware.py        #   Adaptive scheduling
│   │   └── reverse_scheduler.py #   Backward pass
│   ├── loader/                  # Model loaders
│   │   ├── huggingface.py       #   HuggingFace downloader + parser
│   │   └── safetensors_storage.py
│   └── executors/               # Model-specific executors
│       ├── _common.py           #   Shared math (RoPE, RMSNorm, SiLU, ...)
│       ├── gpt2.py              #   GPT-2
│       ├── llama.py             #   LLaMA
│       ├── glm4.py              #   GLM-4
│       ├── glm4_moe.py          #   GLM-4 MoE
│       └── qwen3_next.py        #   Qwen3-Coder-Next
├── examples/
│   ├── dry_run.py               # Mock inference (no download)
│   └── demo_inference.py        # Full text generation demo
├── tests/                       # 201 tests (pytest)
└── docs/
    └── offloading_runtime_design.md
```

## Testing

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run specific test
uv run python -m pytest tests/test_integration_inference.py -v

# Run a single test class
uv run python -m pytest tests/test_integration_inference.py::TestQwen3NextEndToEnd -v
```

Current status: **201 passed, 15 skipped** (skips are for missing GPU backends)

## How It Works

1. **Model Loading**: HuggingFaceLoader가 config.json을 읽어 아키텍처를 감지하고, safetensors 파일에서 텐서 메타데이터를 수집하여 레이어별 LayerSpec을 생성합니다.

2. **Weight Streaming**: 각 레이어의 가중치를 SafetensorsStorage에서 호스트 메모리로 읽고, 비동기 H2D 복사로 GPU에 전송합니다. Scheduler가 다음 레이어를 미리 프리페치하여 전송과 연산이 오버랩됩니다.

3. **Layer Execution**: Executor가 GPU 메모리의 가중치를 사용해 레이어별 순전파를 수행합니다. 완료된 레이어의 가중치는 즉시 해제됩니다.

4. **Token Generation**: 모든 레이어를 통과한 hidden state에 LM head를 적용하여 다음 토큰의 logits를 계산하고, greedy decoding으로 토큰을 선택합니다.

## License

MIT
