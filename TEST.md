# 테스트 가이드

## 테스트 실행 방법

**중요**: 반드시 `--extra dev` 플래그를 사용해야 프로젝트 가상환경에서 pytest가 실행됩니다.

```bash
# 전체 테스트 실행 (권장)
uv run --extra dev pytest tests/ -v

# 간결한 출력
uv run --extra dev pytest tests/ -q

# 특정 모듈만 테스트
uv run --extra dev pytest tests/test_runtime.py -v
uv run --extra dev pytest tests/test_executor_np.py -v
```

> **참고**: `uv run pytest`로 실행하면 별도의 도구 환경에서 pytest가 실행되어
> numpy 등 프로젝트 의존성을 찾지 못해 대부분의 테스트가 skip됩니다.

## 현재 상태

- **245 passed, 15 skipped** (GPU 백엔드 미설치 환경 기준)
- 15개 skip: CUDA, ROCm, MPS 백엔드 관련 테스트 (해당 하드웨어/드라이버 필요)

## 테스트 범위

| 테스트 파일 | 대상 |
|------------|------|
| `test_runtime.py` | OffloadRuntime 핵심 루프, 메트릭, 버퍼 풀 |
| `test_executor_np.py` | NumPy 수학 함수, 모델별 executor |
| `test_executor.py` | PassthroughExecutor |
| `test_scheduler.py` | LookaheadScheduler |
| `test_block_scheduler.py` | BlockScheduler |
| `test_cost_aware_scheduler.py` | CostAwareScheduler |
| `test_resource_aware_scheduler.py` | ResourceAwareScheduler |
| `test_reverse_scheduler.py` | ReverseLookaheadScheduler |
| `test_storage.py` | ShardedMMapStorage |
| `test_safetensors_storage.py` | SafetensorsStorage |
| `test_backends.py` | NullBackend + CUDA |
| `test_rocm_backend.py` | ROCmBackend |
| `test_mps_backend.py` | MPSBackend |
| `test_buffer_pool.py` | DeviceBufferPool |
| `test_pinned_pool.py` | PinnedHostBufferPool |
| `test_quantize.py` | INT8/FP16/BFloat16 Dequantizer |
| `test_training.py` | TrainingRuntime (LoRA) |
| `test_huggingface_loader.py` | HuggingFace 모델 로더 |
| `test_integration_inference.py` | 엔드-투-엔드 추론 |
| `test_exports.py` | 공개 API 계약 |

## GPU 백엔드 테스트

| 환경 | 백엔드 | 설치 |
|------|--------|------|
| NVIDIA GPU | CUDABackend | `uv pip install cuda-python` |
| AMD GPU | ROCmBackend | `uv pip install hip-python` |
| Apple Silicon | MPSBackend | `uv pip install pyobjc-framework-Metal` |

## 지원 모델

| 모델 | 크기 | 용도 |
|------|------|------|
| GPT-2 small | ~500MB | 가장 가볍고 검증 용이 |
| LLaMA 3.2 1B | ~2.2GB | VRAM 제한 시나리오 테스트 |
| GLM-4 9B | ~18GB | 실제 오프로딩 케이스 |
| GLM-4.7 MoE | ~159GB | MoE 아키텍처 테스트 |
| Qwen3-Coder-Next | ~159GB | 하이브리드 아키텍처 테스트 |
