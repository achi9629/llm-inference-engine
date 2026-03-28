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
