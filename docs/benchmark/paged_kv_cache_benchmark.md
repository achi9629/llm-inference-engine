# Paged KV Cache Benchmark — Day 16 (Block-Level Memory Management)

## Setup

| Parameter          | Value                                                                  |
|--------------------|------------------------------------------------------------------------|
| Model              | GPT-2 124M (custom implementation)                                     |
| Parameters         | 124,439,808 (160 state_dict keys)                                      |
| Precision          | fp32                                                                   |
| GPU                | NVIDIA A100-SXM4-80GB                                                  |
| Peak TFLOPS (fp32) | 19.5                                                                   |
| PyTorch            | 2.4.0+cu121                                                            |
| Python             | 3.10.18                                                                |
| Standard KV Cache  | Pre-allocated per-layer `(B, n_heads, n_ctx, head_dim)`                |
| Paged KV Cache     | Block pool `(num_blocks, n_layers, 2, block_size, n_heads, head_dim)`  |
| Batching           | Static batching with left padding                                      |
| Sampling           | Greedy (argmax)                                                        |

---

## 1. Peak GPU Memory vs Num Blocks

**Goal:** Compare peak GPU memory between standard and paged KV cache as the block pool size varies.

**Config:** Fixed `batch_size=4`, `block_size=16`, `max_tokens=50`. Standard uses `max_tokens_for_kv_cache = n_ctx = 1024`. Six batches of 4 prompts each (24 prompts total).

### Standard KV Cache

| Total Tokens | Latency (s) | Tok/s | Peak Mem (MB) | GPU Util (%) | MFU (%) |
|--------------|-------------|-------|---------------|--------------|---------|
| 200          | 0.359       | 556.6 | 966.8         | 19.0         | 0.71    |
| 200          | 0.362       | 552.9 | 966.8         | 31.0         | 0.71    |
| 200          | 0.347       | 577.1 | 966.8         | 31.0         | 0.74    |
| 200          | 0.355       | 563.0 | 966.8         | 31.0         | 0.72    |
| 200          | 0.348       | 574.6 | 966.8         | 31.0         | 0.73    |
| 200          | 0.354       | 565.5 | 966.8         | 31.0         | 0.72    |

### Paged KV Cache (num_blocks=64)

| Total Tokens | Latency (s) | Tok/s | Peak Mem (MB) | GPU Util (%) | MFU (%) |
|--------------|-------------|-------|---------------|--------------|---------|
| 200          | 0.462       | 433.1 | 620.2         | 30.0         | 0.55    |
| 200          | 0.461       | 434.0 | 619.4         | 30.0         | 0.55    |
| 200          | 0.459       | 435.7 | 620.2         | 31.0         | 0.56    |
| 200          | 0.458       | 436.8 | 619.4         | 30.0         | 0.56    |
| 200          | 0.458       | 436.4 | 620.2         | 30.0         | 0.56    |
| 200          | 0.459       | 435.3 | 619.4         | 31.0         | 0.56    |

### Comparison Across num_blocks

| num_blocks | Standard Peak (MB) | Paged Peak (MB) | Memory Savings | Standard Tok/s | Paged Tok/s |
|------------|--------------------|-----------------|----------------|----------------|-------------|
| 64         | 966.8              | 620.2           | **35.9%**      | ~565           | ~435        |
| 128        | 972.7              | 692.2           | **28.8%**      | ~586           | ~410        |
| 256        | 972.7              | 836.2           | **14.0%**      | ~583           | ~436        |

**Observations:**

- **Standard memory is constant** (~970 MB) — pre-allocates `[batch, heads, n_ctx=1024, d_head]` upfront regardless of `num_blocks`
- **Paged memory scales linearly** with `num_blocks` — only allocates blocks on demand within the pre-allocated pool
- At equal logical capacity (64 blocks × 16 = 1024 tokens = `n_ctx`), paged saves ~36% because it only fills blocks actually touched during generation (50 max_tokens << 1024 capacity)
- **Paged throughput gap narrowed to ~1.34x** after vectorizing `update_cache()` — batched writes (`index_select` instead of per-token loops) and batched reads (single `index_select` instead of per-sequence loops) reduced kernel launch overhead. Previously ~1.9x slower with Python-level per-token loops

![Memory: Standard vs Paged by Num Blocks](../../assets/plots/paged_memory_vs_blocks.png)

---

## 2. Peak GPU Memory vs Batch Size

**Goal:** Measure how memory and throughput scale with batch size for both cache types.

**Config:** Fixed `num_blocks=1900`, `block_size=16`, `max_tokens=50`. Varying `batch_size` from 1 to 256. Single timed run per batch size.

### Standard KV cache

| Batch Size | Total Tokens | Latency (s) | Tok/s    | Peak Mem (MB) | GPU Util (%) | MFU (%) |
|------------|--------------|-------------|----------|---------------|--------------|---------|
| 1          | 50           | 0.326       | 153.5    | 642.8         | 21.0         | 0.20    |
| 2          | 100          | 0.335       | 298.4    | 750.8         | 34.0         | 0.38    |
| 4          | 200          | 0.335       | 597.5    | 966.8         | 33.0         | 0.76    |
| 8          | 400          | 0.362       | 1,104.9  | 1,398.8       | 33.0         | 1.41    |
| 16         | 800          | 0.369       | 2,168.3  | 2,262.8       | 39.0         | 2.77    |
| 32         | 1,600        | 0.394       | 4,063.7  | 3,990.8       | 36.0         | 5.19    |
| 64         | 3,200        | 0.454       | 7,042.4  | 7,446.8       | 45.0         | 8.99    |
| 128        | 6,400        | 0.578       | 11,078.7 | 14,358.8      | 48.0         | 14.14   |
| 256        | 12,800       | 0.837       | 15,299.8 | 28,182.8      | 52.0         | 19.53   |

### Paged KV Cache

| Batch Size | Total Tokens | Latency (s) | Tok/s   | Peak Mem (MB) | GPU Util (%) | MFU (%) |
|------------|--------------|-------------|---------|---------------|--------------|---------|
| 1          | 50           | 0.440       | 113.7   | 2,681.2       | 26.0         | 0.15    |
| 2          | 100          | 0.452       | 221.3   | 2,683.6       | 31.0         | 0.28    |
| 4          | 200          | 0.465       | 430.1   | 2,687.2       | 30.0         | 0.55    |
| 8          | 400          | 0.484       | 826.1   | 2,695.4       | 31.0         | 1.05    |
| 16         | 800          | 0.525       | 1,523.1 | 2,712.0       | 32.0         | 1.94    |
| 32         | 1,600        | 0.581       | 2,752.2 | 2,746.7       | 32.0         | 3.51    |
| 64         | 3,200        | 0.710       | 4,506.8 | 2,813.6       | 40.0         | 5.75    |
| 128        | 6,400        | 1.007       | 6,355.5 | 2,949.2       | 42.0         | 8.11    |
| 256        | 12,800       | 1.513       | 8,460.7 | 3,220.3       | 49.0         | 10.80   |

### Side-by-Side Comparison

| Batch Size | Std Mem (MB) | Paged Mem (MB) | Winner       | Std Tok/s | Paged Tok/s | Slowdown |
|------------|--------------|----------------|--------------|-----------|-------------|----------|
| 1          | 642.8        | 2,681.2        | Standard     | 153.5     | 113.7       | 1.4x     |
| 4          | 966.8        | 2,687.2        | Standard     | 597.5     | 430.1       | 1.4x     |
| 16         | 2,262.8      | 2,712.0        | Standard     | 2,168.3   | 1,523.1     | 1.4x     |
| **32**     | **3,990.8**  | **2,746.7**    | **Paged**    | 4,063.7   | 2,752.2     | 1.5x     |
| 64         | 7,446.8      | 2,813.6        | Paged (2.6x) | 7,042.4   | 4,506.8     | 1.6x     |
| 128        | 14,358.8     | 2,949.2        | Paged (4.9x) | 11,078.7  | 6,355.5     | 1.7x     |
| 256        | 28,182.8     | 3,220.3        | Paged (8.7x) | 15,299.8  | 8,460.7     | 1.8x     |

**Observations:**

- **Memory crossover at batch_size ≈ 24–32.** Below that, standard wins because paged pre-allocates the entire block pool (~2,138 MB for 1900 blocks) upfront regardless of usage
- **Standard memory grows linearly** — ~108 MB per additional sequence (`n_ctx × n_layers × n_heads × d_head × 2 × 4 bytes`)
- **Paged memory is nearly flat** — only 539 MB growth from batch=1 to batch=256 (activation tensors scaling, not KV blocks). The block pool is shared and fixed
- **Both cache types now scale throughput with batch size** — standard scales 100x (153 → 15,300 tok/s), paged scales 74x (114 → 8,461 tok/s). After vectorizing `update_cache()` with batched `index_select` reads and advanced-indexing writes, paged throughput no longer plateaus
- **Throughput slowdown is now a consistent ~1.4–1.8x** across all batch sizes — the remaining gap is the inherent cost of block indirection (`index_select` + padding + slicing) vs contiguous tensor reads. Previously, the slowdown ballooned from 1.3x to 24.6x due to Python-level per-sequence loops in `update_cache()`
- **The value proposition kicks in at scale** — at batch_size ≥ 32, paged uses dramatically less memory (up to 8.7x at batch=256) while maintaining competitive throughput

![Memory: Standard vs Paged by Batch Size](../../assets/plots/paged_memory_vs_batch.png)

![Throughput: Standard vs Paged by Batch Size](../../assets/plots/paged_throughput_vs_batch.png)

---

## 3. Peak GPU Memory vs Sequence Length

**Goal:** Measure how memory and throughput change with generation length for both cache types.

**Config:** Fixed `batch_size=4`, `num_blocks=256`, `block_size=16`. Varying `max_tokens` from 10 to 1000.

### Standard KV CACHE

| Max Tokens | Total Tokens | Latency (s) | Tok/s | Peak Mem (MB) | GPU Util (%) | MFU (%) |
|------------|--------------|-------------|-------|---------------|--------------|---------|
| 10         | 40           | 0.074       | 542.5 | 966.8         | 3.0          | 0.69    |
| 50         | 200          | 0.339       | 590.1 | 966.8         | 32.0         | 0.75    |
| 100        | 400          | 0.684       | 584.5 | 966.8         | 33.0         | 0.75    |
| 200        | 800          | 1.342       | 596.3 | 966.8         | 34.0         | 0.76    |
| 300        | 1,200        | 1.993       | 602.0 | 966.8         | 35.0         | 0.77    |
| 400        | 1,600        | 2.672       | 598.9 | 966.8         | 35.0         | 0.76    |
| 500        | 2,000        | 3.332       | 600.3 | 966.8         | 35.0         | 0.77    |
| 600        | 2,400        | 4.008       | 598.7 | 966.8         | 36.0         | 0.76    |
| 700        | 2,800        | 4.668       | 599.8 | 966.8         | 37.0         | 0.77    |
| 800        | 3,200        | 5.343       | 599.0 | 966.8         | 37.0         | 0.76    |
| 900        | 3,600        | 6.085       | 591.7 | 966.8         | 37.0         | 0.76    |
| 1000       | 4,000        | 6.766       | 591.2 | 966.8         | 37.0         | 0.75    |

### Paged KV Cache (Vectorized)

| Max Tokens | Total Tokens | Latency (s) | Tok/s | Peak Mem (MB) | GPU Util (%) | MFU (%) |
|------------|--------------|-------------|-------|---------------|--------------|---------|
| 10         | 40           | 0.096       | 417.0 | 836.3         | 0.0          | 0.53    |
| 50         | 200          | 0.470       | 425.5 | 836.3         | 30.0         | 0.54    |
| 100        | 400          | 0.941       | 425.2 | 836.3         | 30.0         | 0.54    |
| 200        | 800          | 1.877       | 426.3 | 841.6         | 31.0         | 0.54    |
| 300        | 1,200        | 2.820       | 425.5 | 845.6         | 32.0         | 0.54    |
| 400        | 1,600        | 3.776       | 423.8 | 849.1         | 33.0         | 0.54    |
| 500        | 2,000        | 4.739       | 422.0 | 853.6         | 35.0         | 0.54    |
| 600        | 2,400        | 5.694       | 421.5 | 861.6         | 36.0         | 0.54    |
| 700        | 2,800        | 6.670       | 419.8 | 863.4         | 38.0         | 0.54    |
| 800        | 3,200        | 7.632       | 419.3 | 868.8         | 39.0         | 0.54    |
| 900        | 3,600        | 8.609       | 418.2 | 872.7         | 41.0         | 0.53    |
| 1000       | 4,000        | 9.600       | 416.7 | 877.3         | 42.0         | 0.53    |

### Side-by-Side Comparison for both

| Max Tokens | Std Mem (MB) | Paged Mem (MB) | Savings | Std Tok/s | Paged Tok/s | Slowdown |
|------------|--------------|----------------|---------|-----------|-------------|----------|
| 10         | 966.8        | 836.3          | 13.5%   | 542.5     | 417.0       | 1.3x     |
| 100        | 966.8        | 836.3          | 13.5%   | 584.5     | 425.2       | 1.4x     |
| 500        | 966.8        | 853.6          | 11.7%   | 600.3     | 422.0       | 1.4x     |
| 1000       | 966.8        | 877.3          | 9.3%    | 591.2     | 416.7       | 1.4x     |

**Observations:**

- **Both lines are nearly flat — for opposite reasons:**
  - **Standard:** Pre-allocates `n_ctx=1024` upfront. Constant at 966.8 MB
  - **Paged:** Pre-allocates the block pool (256 × 16 slots) upfront. Growing from 10→1000 tokens adds only ~41 MB (activations/intermediates)
- **Paged throughput is now nearly constant** (~417–426 tok/s) across all sequence lengths — vectorized `update_cache()` with batched `index_select` reads eliminated the O(seq_len) Python-level overhead that previously caused degradation from 286→127 tok/s
- **Throughput slowdown is a consistent ~1.3–1.4x** regardless of sequence length — the remaining gap is block indirection overhead (`index_select` + padding), not Python loops
- **Memory savings are modest** (9–13%) at `batch_size=4` because the paged pool (256 blocks) is nearly as large as standard's pre-allocation. Paged shines at larger batch sizes (see Benchmark 2)

![Memory & Throughput vs Sequence Length](../../assets/plots/paged_memory_vs_seqlen.png)

---

## 4. Internal Fragmentation vs Block Size

**Goal:** Measure how block size affects wasted KV slots, throughput, and memory.

**Config:** Fixed `batch_size=4`, `max_tokens=50`, `total_capacity=4096` (num_blocks × block_size held constant). Paged cache only — standard doesn't use blocks.

| Block Size | Num Blocks | Blks/Seq | Alloc Slots | Wasted | Frag (%) | Tok/s | Latency (s) | Peak Mem (MB) |
|------------|------------|----------|-------------|--------|----------|-------|-------------|---------------|
| 4          | 1,024      | 13       | 52          | 2      | **3.8**  | 415.3 | 0.481       | 830.3         |
| 8          | 512        | 7        | 56          | 6      | 10.7     | 431.9 | 0.463       | 830.3         |
| 16         | 256        | 4        | 64          | 14     | 21.9     | 433.9 | 0.461       | 830.3         |
| 32         | 128        | 2        | 64          | 14     | 21.9     | 437.9 | 0.457       | 830.3         |
| 64         | 64         | 1        | 64          | 14     | 21.9     | 442.8 | 0.452       | 830.7         |

**Observations:**

- **Memory is constant** (~830 MB) — `total_capacity=4096` is fixed. Whether 1024 tiny blocks or 64 large blocks, the total KV tensor volume is the same
- **Fragmentation decreases with smaller blocks:** `block_size=4` wastes only 2 slots per sequence (3.8%), while `block_size≥16` wastes 14 slots (21.9%). Note: `block_size=16/32/64` all waste the same 14 slots because `50 mod 16 = 50 mod 32 = 50 mod 64 = 14`
- **Throughput spread across block sizes is now only ~6%** (416–443 tok/s) — vectorized `update_cache()` with batched `index_select` eliminated the Python-level per-block lookups that previously made `block_size=4` 35% slower than `block_size=64` (231 vs 312 tok/s). Now `block_size=4` is only 6% slower (416 vs 443 tok/s)
- **Sweet spot: `block_size=16`** — still the best balance. Fragmentation (21.9%) is the same as larger blocks, throughput (434 tok/s) is within 2% of the fastest, and smaller blocks enable better memory utilization for variable-length workloads. This is vLLM's default for the same reason

![Fragmentation vs Block Size](../../assets/plots/paged_frag_vs_blocksize.png)

### The Granularity Trade-off

| Block Size | Fragmentation        | Throughput          | Verdict                                       |
|------------|----------------------|---------------------|-----------------------------------------------|
| 4          | Best (3.8%)          | 416 tok/s (−6%)     | Lowest fragmentation, minimal throughput cost |
| **16**     | **Moderate (21.9%)** | **434 tok/s (−2%)** | **Sweet spot**                                |
| 64         | Same (21.9%)         | Best (443 tok/s)    | Over-coarse for variable-length workloads     |

---

## 5. Allocator Pressure (Stress Test)

**Goal:** Find the capacity ceiling and verify clean failure when the block pool is exhausted.

**Config:** Deliberately small pool: `num_blocks=64`, `block_size=16`, `max_tokens=50`. Each sequence needs `ceil(50/16) = 4 blocks`. Total capacity: 64 blocks = 16 sequences max.

| Batch | Blocks Needed | Fits? | Util (%) | Status            | Tok/s   | Latency (s) | Peak Mem (MB) |
|-------|---------------|-------|----------|-------------------|---------|-------------|---------------|
| 1     | 4             | Yes   | 6.2      | OK                | 113.2   | 0.442       | 610.0         |
| 2     | 8             | Yes   | 12.5     | OK                | 219.1   | 0.456       | 611.9         |
| 4     | 16            | Yes   | 25.0     | OK                | 423.4   | 0.472       | 616.0         |
| 8     | 32            | Yes   | 50.0     | OK                | 812.5   | 0.492       | 624.2         |
| 16    | 64            | Yes   | 100.0    | OK                | 1,537.4 | 0.520       | 640.7         |
| 32    | 128           | No    | 200.0    | FAIL: MemoryError | —       | —           | —             |
| 64    | 256           | No    | 400.0    | FAIL: MemoryError | —       | —           | —             |

**Observations:**

- **Allocator fails cleanly at the exact theoretical boundary** — batch_size=16 needs exactly 64 blocks (100% utilization) and succeeds; batch_size=32 needs 128 blocks and raises `MemoryError`
- **No off-by-one, no silent corruption** — the capacity formula `max_batch = num_blocks // ceil(max_tokens / block_size)` = `64 // 4 = 16` holds exactly
- **Throughput now scales properly with batch size** — 113→1,537 tok/s (13.6x) from bs=1 to bs=16, compared to 121→498 tok/s (4.1x) pre-vectorization. At bs=16, throughput improved **3.1x** (498→1,537 tok/s)
- **Latency is nearly flat** across batch sizes (0.44–0.52s) — vectorized `update_cache()` makes per-step cost near-constant regardless of batch size, so latency grows only ~18% while batch grows 16x
- **Memory grows ~31 MB** across the full utilization range (610 → 641 MB) — activation/intermediate tensors scale with batch, not the KV block pool (which is pre-allocated)
- **In production**, this is how admission control works: the scheduler checks `allocator.num_free_blocks() >= blocks_per_seq` before accepting a new request. If insufficient, the request queues until blocks are freed

![Allocator Pressure Test](../../assets/plots/paged_allocator_pressure.png)

---

## 6. Load Stress (Day 18): OOM Boundary Test

**Goal:** Compare how far standard vs paged KV cache can scale under concurrent load before failure.

**Method:**

- Run each cache type in a separate process to avoid post-OOM CUDA allocator contamination.
- Increase batch size geometrically until failure.
- Stop at first failure and record failure type.

**Config:** `max_tokens=50`, `block_size=16`, `num_blocks=4096`, batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]`.

### Side-by-Side Results

| Batch Size | Std Mem (MB) | Std Status | Paged Mem (MB) | Paged Status           | Ratio (Std/Paged) |
|------------|--------------|------------|----------------|------------------------|-------------------|
| 1          | 642.8        | OK         | 5145.9         | OK                     | 0.1               |
| 2          | 750.8        | OK         | 5147.9         | OK                     | 0.1               |
| 4          | 966.8        | OK         | 5151.9         | OK                     | 0.2               |
| 8          | 1398.8       | OK         | 5160.7         | OK                     | 0.3               |
| 16         | 2262.8       | OK         | 5176.7         | OK                     | 0.4               |
| 32         | 3990.8       | OK         | 5211.4         | OK                     | 0.8               |
| 64         | 7446.8       | OK         | 5278.4         | OK                     | 1.4               |
| 128        | 14358.8      | OK         | 5414.5         | OK                     | 2.7               |
| 256        | 28182.8      | OK         | 5685.9         | OK                     | 5.0               |
| 512        | 55830.8      | OK         | 6228.9         | OK                     | 9.0               |
| 1024       | N/A          | OOM        | 7312.4         | OK                     | N/A               |
| 2048       | N/A          | -          | N/A            | Not enough free blocks | N/A               |
| 4096       | N/A          | -          | N/A            | -                      | N/A               |

![Load Stress: Memory vs Batch Size](../../assets/plots/load_stress_memory_vs_batch.png)

![Load Stress: Memory Ratio](../../assets/plots/load_stress_memory_ratio.png)

![Load Stress: Side-by-Side Memory Comparison](../../assets/plots/load_stress_bar_chart.png)

**Observations:**

- **Memory crossover occurs between batch 32 and 64.**
- **Paged memory remains relatively flat** as batch increases (fixed block pool + modest activation growth).
- **At batch 512, standard uses ~9x more memory** than paged (55.8 GB vs 6.2 GB).
- **Standard fails first** at batch 1024 with CUDA OOM.
- **Paged fails later** at batch 2048 due to block-pool exhaustion, not CUDA OOM.

**Conclusion:**

Paged KV cache significantly extends serving capacity under concurrent load. In this setup, paged survives a full additional scaling step beyond standard OOM while using substantially less memory at high batch sizes.

---

## 7. Single-User Latency (Day 18): Cold Start vs Warm

**Goal:** Measure single-request latency floor for standard vs paged KV cache, including cold-start penalty.

**Config:** `batch_size=1`, `max_tokens=50`, `num_blocks=128`, `block_size=16`. Warm = mean of 17 iterations after 3 warmup discards.

> **Note:** Only the first row (standard, prompt_len=32) is a true cold start — CUDA context initialization, kernel JIT, and cuBLAS handle creation happen once per process. All subsequent "cold" values reflect only engine-creation overhead on an already-warm CUDA runtime.

### Results

| Phase | Cache    | Prompt Len | Latency (s) | Tok/s     | Peak Mem (MB) |
|-------|----------|------------|-------------|-----------|---------------|
| Cold  | standard | 32         | 0.494       | 101.3     | 634.7         |
| Warm  | standard | 32         | 0.300±0.001 | 166.6±0.7 | 642.8±0.0     |
| Cold  | paged    | 32         | 0.422       | 118.5     | 687.8         |
| Warm  | paged    | 32         | 0.419±0.002 | 119.3±0.6 | 687.8±0.0     |
| Cold  | standard | 256        | 0.305       | 164.1     | 663.1         |
| Warm  | standard | 256        | 0.302±0.002 | 165.4±0.9 | 662.7±0.5     |
| Cold  | paged    | 256        | 0.431       | 116.0     | 734.2         |
| Warm  | paged    | 256        | 0.427±0.002 | 117.1±0.4 | 734.2±0.0     |
| Cold  | standard | 512        | 0.313       | 159.7     | 715.0         |
| Warm  | standard | 512        | 0.307±0.002 | 162.8±0.9 | 714.6±0.5     |
| Cold  | paged    | 512        | 0.438       | 114.2     | 790.4         |
| Warm  | paged    | 512        | 0.435±0.002 | 114.9±0.4 | 790.4±0.0     |

| Prompt Len | Standard Warm | Paged Warm | Slowdown |
|------------|---------------|------------|----------|
| 32         | 0.300s        | 0.419s     | 1.40x    |
| 256        | 0.302s        | 0.427s     | 1.41x    |
| 512        | 0.307s        | 0.435s     | 1.42x    |

### Observations

- **True cold-start penalty: ~1.65x** (0.494s vs 0.300s) — only visible on the first `generate()` call in the process due to CUDA context initialization and kernel JIT compilation
- **Paged latency overhead is now constant at ~1.4x** across all prompt lengths — vectorized `update_cache()` eliminated the O(seq_len) Python-level scatter/gather that previously caused slowdown to grow from 1.37x → 2.84x. Now 1.40x → 1.42x
- **Standard latency is nearly flat** (~0.30s) across prompt lengths — KV cache makes decode cost O(1) per step
- **Paged latency is also nearly flat** (~0.42–0.44s) — the remaining ~0.13s gap is constant block indirection overhead (`index_select` + padding), not sequence-length-dependent Python loops
- **Variance is extremely tight** — std of 0.001–0.002s confirms stable steady-state
- **Memory difference is minimal at batch_size=1** (~45–76 MB more for paged) — the paged advantage only manifests at higher batch sizes

---

## 8. Summary

### Memory Analysis

| Scenario                  | Standard    | Paged      | Winner                 |
|---------------------------|-------------|------------|------------------------|
| Small batch (bs=4)        | 966.8 MB    | 620–836 MB | Paged (14–36% savings) |
| Large batch (bs=256)      | 28,182.8 MB | 3,220.3 MB | **Paged (8.7x less)**  |
| Short sequences (50 tok)  | 966.8 MB    | 836.2 MB   | Paged (13.5% savings)  |
| Long sequences (1000 tok) | 966.8 MB    | 868.7 MB   | Paged (10.1% savings)  |

### Throughput Analysis

| Scenario          | Standard Tok/s | Paged Tok/s | Paged Slowdown |
|-------------------|----------------|-------------|----------------|
| bs=4, 50 tokens   | ~590           | ~435        | 1.4x           |
| bs=256, 50 tokens | 15,300         | 8,461       | 1.8x           |
| bs=4, 1000 tokens | 591            | 417         | 1.4x           |

### Key Takeaways

1. **Paged KV cache is a memory optimization with competitive throughput** — after vectorizing `update_cache()`, the throughput gap is a consistent ~1.4–1.8x across all scenarios (previously 1.9–24.6x)
2. **Memory crossover at batch_size ≈ 24–32** — below that threshold, standard is more memory-efficient due to paged's upfront pool allocation
3. **At large batch sizes, paged dominates** — 8.7x less memory at bs=256, enabling far more concurrent sequences within a given GPU memory budget
4. **Throughput gap is now constant** — ~1.4x at small batch/sequence, ~1.8x at bs=256. The remaining overhead is inherent block indirection (`index_select` + padding), not Python loops. Production systems (vLLM, TGI) close this further with fused CUDA kernels (`paged_attention_v1/v2`)
5. **`block_size=16` is the sweet spot** — balances fragmentation (21.9%) against lookup overhead (only 2% slower than the coarsest blocks after vectorization)
6. **Allocator fails cleanly** — `MemoryError` at the exact capacity boundary, enabling reliable admission control
7. **Single-user paged latency overhead: constant ~1.4x** — no longer scales with prompt length (was 1.4–2.8x pre-vectorization). Variance is extremely tight (±0.002s)

### Paged KV Cache Memory Breakdown (estimated, batch_size=4)

| Component                                                                     | Size (MB) |
|-------------------------------------------------------------------------------|-----------|
| Model weights (124M × 4 bytes)                                                | ~497      |
| KV block pool (256 blocks × 16 × 12 layers × 2 × 12 heads × 64 dim × 4 bytes) | ~339      |
| Activations + tokenizer + overhead                                            | ~30       |
| **Total**                                                                     | **~836**  |

---

## Benchmark Progression

| Day | Feature                   | Tok/s (best)       | Peak Mem        | Improvement               |
|-----|---------------------------|--------------------|-----------------|---------------------------|
| 7   | Baseline (no cache)       | 169 tok/s          | 540–930 MB      | —                         |
| 9   | + KV Cache                | 174 tok/s          | 643.8 MB        | 1.03x, constant memory    |
| 11  | + Batching       (bs=256) | 15,299 tok/s       | 28,182 MB       | 118x (from bs=1)          |
| 13  | + Continuous Batching     | 591 tok/s (bs=4)   | 1,399 MB        | Same as static (expected) |
| 16  | + Paged KV Cache (bs=256) | 8,460 tok/s        | **3,220 MB**    | **8.7x less memory**      |

**Next:** Day 17 — Serving Layer (FastAPI server, request handler, router)
