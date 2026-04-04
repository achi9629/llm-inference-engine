# Design Decisions

## 1. Block Ownership: Engine vs Scheduler

**Problem:** Both `InferenceEngine` and `ContinuousBatchingScheduler` can manage paged KV cache blocks (allocate/free via `BlockTable`). When combined through the Router, both would try to allocate blocks — causing double allocation.

**Our Choice:** Dual ownership by path.

| Path                            | Block Owner | Why                                                      |
|---------------------------------|-------------|----------------------------------------------------------|
| Offline (benchmarks, scripts)   | Engine      | `engine.generate()` works standalone without a scheduler |
| Online (HTTP → Router → Server) | Engine      | Scheduler tracks request state only, avoids conflict     |

**How vLLM does it:** Scheduler owns blocks exclusively. Engine never allocates — it assumes blocks are pre-allocated. This enables memory-aware admission control ("reject request if not enough blocks") and preemption ("evict running request to free blocks").

**Tradeoff:**

- Our approach: simpler, both paths work independently, no refactoring needed
- vLLM approach: architecturally cleaner for serving, but breaks standalone `engine.generate()`

**Future refactor:** Add `external_block_management` flag to InferenceEngine. When True, engine skips block allocation (scheduler owns). When False (default), engine manages blocks (standalone usage).

## 2. Standard KV Cache: Fixed Batch Size (batch_size=1 for Serving)

**Problem:** `KVCache` pre-allocates tensors of shape `(batch_size, n_head, max_seq_len, head_dim)` at init. The `update_cache` method writes directly into this fixed-size buffer:

```python
self.k_cache[layer_idx][:, :, start:end, :] = k   # k shape must match batch dim
```

When the Router's `_generation_loop` batches N requests (where N varies per scheduling step), passing N prompts to `engine.generate()` produces K/V tensors with `batch_dim=N`. If `N != batch_size`, PyTorch raises:

```bash
RuntimeError: The expanded size of the tensor (32) must match the existing size (5)
at non-singleton dimension 0
```

This means standard KVCache **cannot handle dynamic batch sizes** — the exact number of requests must equal the pre-allocated `batch_size` at every step.

**Our Choice:** Run standard cache with `batch_size=1` (sequential, no batching) in serving mode.

| Cache Type | Serving Mode     | batch_size    | Dynamic Batching | Why                                |
|------------|------------------|---------------|------------------|------------------------------------|
| Standard   | Sequential       | 1             | No               | Fixed tensor shape, no reshape     |
| Paged      | Batched (async)  | max_batch_size| Yes              | Per-sequence blocks, no batch dim  |

**Why paged cache enables dynamic batching:** Paged KV cache stores K/V at the per-sequence, per-block level — there is no batch dimension in the underlying storage. `PagedCacheContext` wraps the paged cache with a list of sequence IDs, so any number of sequences can be processed together without shape conflicts. The block pool is pre-allocated once, and individual blocks are allocated/freed per-sequence.

**Why not recreate KVCache per batch?** We could `del self.kv_cache` and reinitialize with the actual batch size at each `generate()` call. This would work but:

- Re-allocates GPU memory every call (CUDA malloc overhead)
- No advantage over paged cache, which already solves this cleanly
- Standard cache exists as a simpler baseline for comparison, not as a production path

**Benchmark evidence (GPT-2 124M, A100, max_tokens=50):**

| Metric (c=64)     | Standard (sequential) | Paged (batched) | Speedup |
|-------------------|-----------------------|-----------------|---------|
| Short throughput  | 165 tok/s             | 532 tok/s       | 3.2x    |
| Medium throughput | 166 tok/s             | 455 tok/s       | 2.7x    |
| Long throughput   | 167 tok/s             | 234 tok/s       | 1.4x    |
| Short p50 latency | 19.4s                 | 6.0s            | 3.2x    |

Standard throughput is flat (~165 tok/s) regardless of concurrency — requests queue and process one at a time. Paged throughput scales with concurrency until GPU compute saturates (~c=64).

**Takeaway:** Standard KV cache is a correct, simple baseline. Paged KV cache is required for dynamic batching in production serving. This is the same fundamental reason why vLLM, TGI, and other production systems use paged/block-level memory management.
