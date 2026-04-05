# Load Test Benchmark — Day 19 (Concurrent HTTP Load Testing)

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
| Server             | FastAPI + Uvicorn on `127.0.0.1:8000`                                  |
| Client             | `httpx.AsyncClient` with `asyncio.gather` (trust_env=False)            |
| Standard Config    | `batch_size=1`, sequential (no `--async_mode`), KV cache enabled       |
| Paged Config       | `max_batch_size=32`, `--async_mode`, `num_blocks=512`, `block_size=16` |
| Max Tokens         | 50 per request                                                         |
| Concurrency Levels | 1, 4, 8, 16, 32, 64, 128                                               |

**Prompts:**

| Type   | Tokens (~) | Text                                                                                                                                                                          |
|--------|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Short  | 6          | "The capital of France is"                                                                                                                                                    |
| Medium | 28         | "Deep learning has revolutionized the field of artificial intelligence by enabling models to learn complex patterns from large datasets without explicit feature engineering" |
| Long   | 56         | "The theory of relativity, proposed by Albert Einstein in the early twentieth century, fundamentally changed our understanding of space, time, and gravity..."                |

**Key difference:** Standard cache uses `batch_size=1` (sequential processing) because its fixed-shape tensors `(B, n_head, max_seq_len, head_dim)` cannot handle dynamic batch sizes. Paged cache uses `--async_mode` with continuous batching — see [design_decisions.md](../concepts/design_decisions.md#2-standard-kv-cache-fixed-batch-size-batch_size1-for-serving).

---

## 1. Aggregate Throughput (tok/s) vs Concurrency

### Standard Cache (Sequential)

| Concurrency | Short tok/s | Medium tok/s | Long tok/s |
|:-----------:|:-----------:|:------------:|:----------:|
| 1           | 158         | 157          | 159        |
| 4           | 165         | 163          | 166        |
| 8           | 165         | 164          | 165        |
| 16          | 166         | 164          | 166        |
| 32          | 166         | 166          | 167        |
| 64          | 165         | 166          | 167        |
| 128         | 166         | 165          | 166        |

### Paged Cache (Batched)

| Concurrency | Short tok/s | Medium tok/s | Long tok/s |
|:-----------:|:-----------:|:------------:|:----------:|
| 1           | 110         | 107          | 106        |
| 4           | 418         | 400          | 394        |
| 8           | 769         | 758          | 714        |
| 16          | 1,385       | 1,341        | 1,190      |
| 32          | 1,446       | 1,002        | 1,306      |
| 64          | **1,840**   | **1,749**    | **1,582**  |
| 128         | 1,659       | 1,442        | 1,023      |

### Side-by-Side (Peak at c=64)

| Prompt | Standard (c=64) | Paged (c=64)  | Speedup   |
|--------|-----------------|---------------|-----------|
| Short  | 165 tok/s       | 1,840 tok/s   | **11.1x** |
| Medium | 166 tok/s       | 1,749 tok/s   | **10.5x** |
| Long   | 167 tok/s       | 1,582 tok/s   | **9.5x**  |

**Observations:**

- Standard throughput is **completely flat** (~165 tok/s) regardless of concurrency — requests are serialized due to `batch_size=1`
- Paged throughput **scales aggressively with concurrency** — vectorized cache operations (batched `index_select` reads, advanced-indexing writes) enable massive GPU parallelism
- Peak throughput at c=64 for all prompt types; decline at c=128 due to scheduling overhead and batch cycling
- **Long prompts now achieve comparable speedups** (9.5x vs 11.1x for short) — vectorized `update_cache()` eliminates the per-sequence scatter bottleneck that previously limited long-prompt batching to 1.4x
- Medium at c=32 shows a throughput dip (1,002 tok/s) — a batch cycling boundary artifact where 32 requests exactly match `max_batch_size=32`, reducing overlap between batches

![Throughput: Standard vs Paged by Concurrency](../../assets/plots/load_throughput_vs_concurrency.png)

---

## 2. Latency Distribution

### p50 Latency (seconds)

| Concurrency | Std Short | Paged Short | Std Medium | Paged Medium | Std Long | Paged Long |
|:-----------:|:---------:|:-----------:|:----------:|:------------:|:--------:|:----------:|
| 1           | 0.32      | 0.46        | 0.32       | 0.47         | 0.31     | 0.47       |
| 4           | 1.21      | 0.48        | 1.22       | 0.50         | 1.20     | 0.51       |
| 8           | 2.42      | 0.52        | 2.43       | 0.53         | 2.42     | 0.56       |
| 16          | 4.83      | 0.57        | 4.88       | 0.59         | 4.81     | 0.67       |
| 32          | 9.66      | 1.09        | 9.64       | 1.14         | 9.58     | 1.22       |
| 64          | 19.36     | 1.73        | 9.81       | 1.82         | 19.14    | 2.00       |
| 128         | 19.51     | 2.23        | 30.40      | 2.54         | 19.30    | 3.48       |

### Tail Latency — Short Prompt

| Concurrency | Std p95 (s) | Paged p95 (s) | Std p99 (s) | Paged p99 (s) |
|:-----------:|:-----------:|:-------------:|:-----------:|:-------------:|
| 1           | 0.32        | 0.46          | 0.32        | 0.46          |
| 8           | 2.42        | 0.52          | 2.42        | 0.52          |
| 32          | 9.66        | 1.10          | 9.66        | 1.10          |
| 64          | 19.37       | 1.73          | 19.37       | 1.74          |
| 128         | 36.66       | 3.80          | 38.17       | 3.81          |

**Observations:**

- **At c=1:** Paged is slightly slower (0.46s vs 0.32s) due to paged memory management overhead (block allocation, scatter/gather)
- **At c≥4:** Paged wins decisively — vectorized batching amortizes the per-request overhead
- **Standard p50 grows linearly** with concurrency: `p50 ≈ concurrency × single_request_latency`. Classic serial queuing behavior
- **Paged p50 stays sub-second up to c=16** (0.57s) and grows much slower than standard — true batching processes multiple requests per GPU forward pass
- **At c=32, short prompt:** Paged delivers **8.9x lower** p50 latency (1.09s vs 9.66s)
- **At c=128:** Paged p99 (3.81s) is **10x lower** than standard p95 (36.66s)

![Latency: p50 by Concurrency](../../assets/plots/load_p50_vs_concurrency.png)

---

## 3. Request Throughput (req/s)

| Concurrency | Std Short | Paged Short | Std Medium | Paged Medium | Std Long | Paged Long |
|:-----------:|:---------:|:-----------:|:----------:|:------------:|:--------:|:----------:|
| 1           | 3.17      | 2.20        | 3.13       | 2.14         | 3.17     | 2.12       |
| 4           | 3.30      | 8.36        | 3.26       | 8.00         | 3.33     | 7.87       |
| 8           | 3.31      | 15.38       | 3.29       | 15.16        | 3.31     | 14.28      |
| 16          | 3.31      | 27.71       | 3.28       | 26.82        | 3.32     | 23.80      |
| 32          | 3.31      | 28.92       | 3.32       | 20.03        | 3.34     | 26.11      |
| 64          | 3.30      | **36.79**   | 3.32       | **34.98**    | 3.34     | **31.64**  |
| 128         | 3.32      | 33.18       | 3.30       | 28.84        | 3.33     | 20.46      |

**Observations:**

- Standard: flat at ~3.3 req/s (one request at a time, each taking ~0.3s)
- Paged short: scales to **36.79 req/s** (11.1x improvement over standard)
- Paged long: scales to **31.64 req/s** (9.5x improvement) — prompt length gap is now small
- req/s = tok/s ÷ 50 (since max_tokens=50 for all requests)

---

## 4. Failure Rate

**Zero failures across all 1,518 total requests** (7 concurrency levels × 3 prompt types × 2 cache types = 42 runs). The error handling in `_generation_loop` (try/except with `future.set_exception()` and `scheduler.complete_request()`) is working correctly.

---

## 5. Key Takeaways

### Why Paged Cache Enables Dynamic Batching

Standard KVCache pre-allocates tensors of shape `(batch_size, n_head, max_seq_len, head_dim)`. The batch dimension is **fixed at init** — passing N≠batch_size requests causes a RuntimeError. This forces `batch_size=1` in serving mode.

Paged KV cache stores K/V at the per-sequence, per-block level with no batch dimension in the underlying storage. `PagedCacheContext` wraps any number of sequences via a list of sequence IDs, enabling dynamic batch sizes every scheduling step.

### Performance Summary

| Metric                        | Standard (sequential) | Paged (batched)  |
|-------------------------------|-----------------------|------------------|
| Peak short throughput         | 166 tok/s             | 1,840 tok/s      |
| Peak short req/s              | 3.3                   | 36.8             |
| p50 latency at c=32 (short)   | 9.66s                 | 1.09s            |
| Throughput scaling            | Flat                  | Sub-linear       |
| Failure rate                  | 0%                    | 0%               |

### Throughput Scaling Behavior

- **Standard:** `throughput ≈ constant` — adding more concurrent users just increases queue depth, not GPU parallelism
- **Paged:** `throughput ∝ min(concurrency, max_batch_size)` — scales until GPU compute saturates at ~c=64
- **Diminishing returns past c=64:** GPU is fully utilized; additional requests queue behind the active batch

### Prompt Length Impact on Batching Gains

| Prompt | Paged Speedup (c=64) | Reason                                                                                                    |
|--------|----------------------|-----------------------------------------------------------------------------------------------------------|
| Short  | 11.1x                | Short prefill, vectorized batch reads/writes maximize GPU parallel compute                                |
| Medium | 10.5x                | Slightly more prefill work, but vectorized cache eliminates per-sequence overhead                         |
| Long   | 9.5x                 | More attention computation per decode step, but vectorization closes the gap (was 1.4x pre-vectorization) |

---

## Benchmark Progression

| Stage                        | Day | Context | Batch Size | Short tok/s | Key Gain                     |
|------------------------------|-----|---------|------------|-------------|------------------------------|
| Baseline (no cache)          | 7   | Direct  | 1          | 127 tok/s   | —                            |
| + KV Cache                   | 9   | Direct  | 1          | 174 tok/s   | 1.4x (avoid recomputing K/V) |
| + Batch Inference            | 11  | Direct  | 4          | 599 tok/s   | 3.4x (GPU parallelism)       |
| + Continuous Batching        | 13  | Direct  | 4          | 599 tok/s   | Same, better utilization     |
| + Paged KV Cache             | 16  | Direct  | 4          | 301 tok/s   | 0.5x (scatter overhead)      |
| + HTTP Serving (sequential)  | 19  | HTTP    | 1          | 165 tok/s   | —  (new baseline for HTTP)   |
| + HTTP Serving (batched)     | 19  | HTTP    | 32         | 532 tok/s   | 3.2x vs HTTP sequential      |
| + Vectorized paged cache     | 20  | HTTP    | 32         | 1,840 tok/s | 11.1x vs HTTP sequential     |

---

## 6. Backpressure & Arrival Patterns (Paged Cache, 100 Requests)

**Test:** 100 requests, `max_batch_size=4`, paged cache with `--async_mode`, 50 max tokens per request.
**Script:** `benchmarks/load/load_v.py`
**Arrival patterns:**

| Pattern  | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| Burst    | All 100 requests fired simultaneously via `asyncio.gather`                  |
| Steady   | One request every 0.1s (`asyncio.ensure_future` + `asyncio.sleep`)          |
| Poisson  | Exponential inter-arrival times at 10 req/s (`np.random.exponential(0.1)`)  |

### 6.1 Throughput (tok/s)

| Pattern   | Short     | Medium    | Long      |
|:---------:|:---------:|:---------:|:---------:|
| **Burst** | **2,095** | **2019**  | **1,755** |
| Steady    | 469       | 470       | 473       |
| Poisson   | 520       | 515       | 518       |

**Observations:**

- **Burst throughput is 4.4x higher than steady** (2,095 vs 469 for short) — with vectorized cache operations, full batch slots (4/4 always active) maximize GPU parallelism efficiency
- Steady and Poisson converge to ~470–520 tok/s across all prompt lengths — partially filled batches at 10 req/s reduce vectorization gains
- **Long prompts now match short/medium under burst** (1,755 vs 2,092, 1.2x gap) — vectorized `update_cache()` eliminates the per-sequence scatter bottleneck (previously 1.7x gap)
- Bottleneck under burst is pure compute; under steady/poisson it's arrival rate limiting effective batch utilization

![Throughput by Arrival Pattern](../../assets/plots/arrival_throughput_bar.png)

### 6.2 Latency Distribution (seconds)

| Pattern  | Prompt | p50   | p90   | p95   | p99   | Mean  | Std  | Min  | Max   |
|:--------:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-----:|
| Burst    | Short  | 2.35  | 2.37  | 2.37  | 2.37  | 1.84  | 0.76 | 0.71 | 2.37  |
| Steady   | Short  | 0.72  | 0.91  | 0.93  | 0.95  | 0.72  | 0.14 | 0.45 | 0.96  |
| Poisson  | Short  | 0.74  | 0.93  | 0.95  | 0.97  | 0.74  | 0.15 | 0.46 | 1.01  |
| Burst    | Medium | 2.97  | 2.99  | 2.99  | 2.99  | 2.95  | 0.23 | 0.65 | 2.99  |
| Steady   | Medium | 0.73  | 0.92  | 0.94  | 0.96  | 0.72  | 0.14 | 0.46 | 0.96  |
| Poisson  | Medium | 0.75  | 0.93  | 0.96  | 1.00  | 0.75  | 0.14 | 0.46 | 1.00  |
| Burst    | Long   | 2.84  | 2.86  | 2.86  | 2.86  | 2.67  | 0.59 | 0.67 | 2.86  |
| Steady   | Long   | 0.80  | 1.00  | 1.01  | 1.02  | 0.79  | 0.15 | 0.47 | 1.02  |
| Poisson  | Long   | 0.79  | 0.98  | 1.01  | 1.05  | 0.79  | 0.15 | 0.46 | 1.06  |

**Observations:**

- **Burst latency collapsed from ~17s to ~2.4s (short)** — vectorized batch processing drains the 100-request queue 7x faster
- **Burst has tight latency spread** (std ≈ 0.23–0.76s) — all requests queue at t=0 and drain rapidly together
- **Burst p50 ≈ wall time** — all requests experience similar total time (~2.4s short, ~2.9s medium/long)
- **Steady/Poisson have sub-second p50** (0.72–0.80s) — at 10 req/s, most requests process immediately without queuing
- **Long prompt burst latency is comparable to short** (2.84s vs 2.35s) — vectorized cache operations eliminate the long-prompt penalty (previously 29s vs 17s)

![Latency by Arrival Pattern](../../assets/plots/arrival_latency_box.png)

### 6.3 Backpressure

| Pattern  | Prompt | Total | Failed | Wall Time (s) | req/s |
|:--------:|:------:|:-----:|:------:|:-------------:|:-----:|
| Burst    | Short  | 100   | 0      | 2.39          | 41.83 |
| Steady   | Short  | 100   | 0      | 10.57         | 9.46  |
| Poisson  | Short  | 100   | 0      | 9.73          | 10.28 |
| Burst    | Medium | 100   | 0      | 3.00          | 33.33 |
| Steady   | Medium | 100   | 0      | 10.60         | 9.43  |
| Poisson  | Medium | 100   | 0      | 9.70          | 10.31 |
| Burst    | Long   | 100   | 0      | 2.87          | 34.83 |
| Steady   | Long   | 100   | 0      | 10.61         | 9.43  |
| Poisson  | Long   | 100   | 0      | 9.64          | 10.38 |

**Zero failures across all 900 requests (3 patterns × 3 prompts × 100 requests).** Burst wall time dropped from ~17–29s to ~2.4–3.0s (7–10x faster queue drain). Steady/Poisson wall times are dominated by inter-arrival delays (~10s), not compute. The scheduler + paged cache absorbs a full 100-request burst without dropping, timing out, or OOM-ing.

---

## 7. GPU Utilization & Memory

Monitored via `pynvml` at 0.25s intervals during each arrival pattern run.

### 7.1 GPU Compute Utilization (%)

| Pattern  | Prompt | Mean  | Min  | Max  | Samples |
|:--------:|:------:|:-----:|:----:|:----:|:-------:|
| Burst    | Short  | 24.7  | 0    | 27   | 69      |
| Steady   | Short  | 25.8  | 9    | 29   | 70      |
| Poisson  | Short  | 26.1  | 15   | 30   | 73      |
| Burst    | Medium | 24.9  | 0    | 27   | 75      |
| Steady   | Medium | 25.5  | 24   | 28   | 78      |
| Poisson  | Medium | 25.3  | 16   | 28   | 79      |
| Burst    | Long   | 20.9  | 0    | 25   | 116     |
| Steady   | Long   | 21.2  | 8    | 25   | 123     |
| Poisson  | Long   | 21.2  | 13   | 25   | 123     |

**Observations:**

- GPU utilization is **low** (~21-26%) — GPT-2 124M is far too small to saturate an A100
- **Burst shows min=0%** — the first pynvml sample captures the moment before GPU kernels launch
- **Long prompts have lower utilization** (~21%) than short (~25%) despite more FLOPs per step — this is because attention is more memory-bandwidth-bound (reading large KV tensors), so the GPU compute units idle while waiting on memory
- Steady/Poisson have slightly higher utilization than burst — requests arrive throughout the run, keeping the GPU continuously engaged (no idle gap at the start)

### 7.2 GPU Memory Usage (MB)

| Pattern  | Prompt | Mean     | Min      | Max      | Total    |
|:--------:|:------:|:--------:|:--------:|:--------:|:--------:|
| Burst    | Short  | 2,438    | 2,405    | 2,439    | 81,920   |
| Steady   | Short  | 2,439    | 2,439    | 2,439    | 81,920   |
| Poisson  | Short  | 2,439    | 2,439    | 2,439    | 81,920   |
| Burst    | Medium | 2,460    | 2,439    | 2,461    | 81,920   |
| Steady   | Medium | 2,461    | 2,461    | 2,461    | 81,920   |
| Poisson  | Medium | 2,461    | 2,461    | 2,461    | 81,920   |
| Burst    | Long   | 2,565    | 2,461    | 2,567    | 81,920   |
| Steady   | Long   | 2,569    | 2,567    | 2,571    | 81,920   |
| Poisson  | Long   | 2,571    | 2,571    | 2,571    | 81,920   |

**Observations:**

- Memory usage is **minimal** — 2.4-2.6 GB out of 81.9 GB total (3% of A100 capacity)
- **Long prompts use ~130 MB more** than short (2,571 vs 2,439 MB) — the paged KV cache allocates more blocks for longer sequences
- **Burst min < steady min** — first sample captures pre-allocation state before all 100 block tables are set up
- Memory is effectively constant during the run — paged cache pre-allocates block pool, so no dynamic GPU memory growth

![GPU Utilization by Pattern](../../assets/plots/arrival_gpu_util.png)

---

## 8. Combined Takeaways

### Arrival Pattern Impact

| Metric                    | Burst                      | Steady         | Poisson         |
|---------------------------|----------------------------|----------------|-----------------|
| Best for throughput?      | **Yes** (4.4x over steady) | No             | Close to steady |
| Best for latency?         | No (highest p50)           | **Yes** (best) | Middle          |
| Realistic traffic model?  | No (DDoS-like)             | No (synthetic) | **Yes** (web)   |
| Backpressure failures?    | 0                          | 0              | 0               |

### Key Findings

1. **Vectorized batching creates a burst advantage** — burst (2,092 tok/s short) is **4.4x higher** than steady (473 tok/s). With all batch slots always full, vectorized cache operations achieve maximum GPU parallelism. Pre-vectorization, burst and steady were within 5%
2. **Burst completes 100 requests in 2–3 seconds** (was 17–29s) — vectorized `update_cache()` reduces per-step overhead, draining the queue 7–10x faster
3. **Long prompts now match short/medium under burst** — 1,742 vs 2,092 tok/s (1.2x gap, was 1.7x). Vectorization eliminates per-sequence scatter cost that previously penalized longer KV reads
4. **Steady/Poisson spread latency while maintaining sub-second p50** — first requests process immediately (min ≈ 0.45s), later ones queue (max ≈ 1.0s), creating a realistic user experience vs burst's uniform ~2.4s
5. **GPU is massively under-utilized** — 21-26% compute, 3% memory. A100 is designed for models 100x larger than GPT-2 124M
6. **Zero failures under all conditions** — server handles 100 simultaneous requests without errors, timeouts, or OOM
