# disk-mem-off

CUDA-first, vendor-portable memory offloading runtime scaffold.

## Quick start (dry run)

```bash
python3 -m examples.dry_run
```

## Layout

- `docs/offloading_runtime_design.md`: design spec for portable layer-wise offloading.
- `offload_runtime/backends/`: backend abstraction and null backend.
- `offload_runtime/storage/`: storage abstraction and mmap/in-memory adapters.
- `offload_runtime/scheduler/`: lookahead prefetch scheduler.
- `offload_runtime/runtime.py`: core layer-wise offload loop.
