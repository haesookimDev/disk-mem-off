# Memory Offloading Runtime Design (CUDA-first, Vendor-portable)

## 1. Goals
- Run very large models with limited VRAM via layer-wise streaming.
- Keep the runtime core vendor-neutral so CUDA is an adapter, not a hard dependency.
- Start low-level in Python (explicit streams/events/memcpy) and preserve a clean path to ROCm/other backends.

## 2. Non-goals (v1)
- Maximum throughput on multi-node clusters.
- Full framework replacement (PyTorch/JAX still used for kernels where practical).
- End-to-end optimizer implementation for full-parameter training.

## 3. Core Principle
- Trade memory capacity for transfer time:
  - Persist weights in host RAM (or NVMe-backed mmap).
  - Prefetch only the next layer/block to device memory.
  - Compute current layer.
  - Evict finished layer immediately.
- Use overlap to reduce penalty:
  - `Transfer stream` copies next weights while `Compute stream` runs current layer.

## 4. Architecture

```
Storage (RAM/NVMe mmap)
    |
    v
Prefetcher -> Host buffers (prefer pinned) -> H2D stream -> Device buffer pool
                                                      |
                                                      v
                                                Compute stream
                                                      |
                                                      v
                                                 Layer executor
                                                      |
                                                      v
                                                    Evictor
```

## 5. Component Boundaries

### 5.1 Backend adapter (vendor-specific)
- Responsibilities:
  - Device allocation/free.
  - Async memcpy (H2D/D2H).
  - Stream/event creation and synchronization.
- Interface remains stable across vendors:
  - `alloc_device`, `free_device`
  - `copy_h2d_async`, `copy_d2h_async`
  - `create_stream`, `record_event`, `wait_event`, `synchronize_stream`

### 5.2 Storage adapter (vendor-neutral)
- Responsibilities:
  - Layer weight metadata lookup.
  - Host buffer lifecycle (`request`, `wait`, `get`, `release`).
- Implementations:
  - `InMemoryStorage` for tests.
  - `ShardedMMapStorage` for production-like loading from binary shards.

### 5.3 Scheduler (vendor-neutral)
- Start with lookahead prefetch:
  - Warm up first `W` layers.
  - For each layer `i`, prefetch `i + W`.
- Extend later:
  - Block-wise scheduling (group 1-4 layers per transfer unit).
  - Cost-aware scheduling with measured transfer/compute times.

### 5.4 Executor (framework-boundary)
- Receives:
  - `LayerSpec`, activations, device weights, backend stream handle.
- Executes actual layer compute.
- Can be implemented via:
  - PyTorch custom op / module call.
  - Triton kernels.
  - Vendor library wrappers.

## 6. Data Format

Use an index JSON that maps each layer to a shard offset:

```json
{
  "layers": [
    { "layer_id": 0, "name": "layer0", "path": "weights-000.bin", "offset": 0, "nbytes": 1048576 }
  ]
}
```

Design notes:
- Keep layer bytes contiguous in file by execution order to improve readahead.
- Align offsets to page boundaries when possible (4KB+).
- For future quantization, store `dtype` and optional scale metadata per entry.

## 7. Runtime Flow (single batch inference)

1. Build ordered layer list.
2. Prefetch first `W` layers to host buffers.
3. For each layer `i`:
   - Wait host buffer readiness.
   - Allocate device buffer.
   - Enqueue H2D copy on transfer stream.
   - Record transfer event, make compute stream wait.
   - Execute layer compute.
   - Free device buffer and release host buffer.
   - Request prefetch for next lookahead layer.
4. Synchronize streams and return outputs + metrics.

## 8. Performance Controls
- Lookahead window `W`: hide copy latency without exploding host memory.
- Device buffer pool: avoid frequent malloc/free when layer sizes are repetitive.
- Pinned host memory: required for high async transfer throughput.
- Block granularity: transfer one layer vs. small layer groups.
- Quantized transfer: reduce PCIe pressure.

## 9. Portability Strategy
- Keep CUDA symbols out of core runtime and scheduler modules.
- Use backend capability flags:
  - `supports_pinned_host`
  - `supports_peer_to_peer`
  - `supports_graph_capture`
- Add backends independently:
  - `cuda_backend.py` (first)
  - `rocm_backend.py` (second)
  - Optional fallback backend for CPU-only simulation.

## 10. Training Extension Plan
- LoRA first (recommended):
  - Base weights offloaded.
  - Small trainable adapters resident on device.
- Full training later:
  - Activation checkpointing.
  - Grad/optimizer offload.
  - Reverse-order prefetch for backward pass.

## 11. Observability (must-have)
- Per layer:
  - `h2d_ms`, `compute_ms`, `stall_ms`, `bytes`.
- Global:
  - End-to-end latency.
  - Effective transfer bandwidth.
  - Transfer/compute overlap ratio.

## 12. Milestones
- M1: Single-GPU inference, lookahead scheduler, mmap storage, null backend tests.
- M2: CUDA backend with real async memcpy + stream/event sync.
- M3: Buffer pooling, pinned memory pipeline, block-wise scheduling.
- M4: LoRA training offload path.

