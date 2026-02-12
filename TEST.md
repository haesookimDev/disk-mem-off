현재 상태
이 프로젝트는 저수준 메모리 오프로딩 프레임워크입니다. 모델 포맷 로더나 실제 연산 executor가 없기 때문에, 지금 바로 "모델을 넣고 추론 실행"하는 것은 불가능합니다.

현재 테스트할 수 있는 것:


# 전체 테스트 (NullBackend + mock executor)
uv run pytest tests/ -v
이것은 파이프라인 흐름(prefetch → H2D → compute → free)을 검증하지만, 실제 GPU 연산은 하지 않습니다.

실제 모델 추론을 테스트하려면
3가지가 필요합니다:

1. 모델 가중치 (ShardedMMapStorage 포맷)
ShardedMMapStorage가 기대하는 인덱스 JSON:


{
  "layers": [
    {"layer_id": 0, "path": "weights-000.bin", "offset": 0, "nbytes": 4096},
    {"layer_id": 1, "path": "weights-000.bin", "offset": 4096, "nbytes": 4096}
  ]
}
2. 실제 연산을 수행하는 LayerExecutor
현재 PassthroughExecutor는 검증만 하고 activations를 그대로 통과시킵니다. 실제 모델에는 행렬 곱셈 등을 수행하는 executor가 필요합니다.

3. GPU 백엔드
환경	백엔드	설치
NVIDIA GPU	CUDABackend	uv pip install cuda-python
AMD GPU	ROCmBackend	uv pip install hip-python
Apple Silicon	MPSBackend	uv pip install pyobjc-framework-Metal
추천하는 테스트 모델
모델	크기	용도
GPT-2 small	~500MB	가장 가볍고 검증 용이
TinyLlama-1.1B	~2.2GB	VRAM 제한 시나리오 테스트에 적합
Llama-2-7B	~14GB	실제 오프로딩이 필요한 현실적 케이스
하지만 이 모델들을 사용하려면 safetensors/GGUF → shard format 변환기와 실제 LayerExecutor 구현이 먼저 필요합니다.