# Baseline Benchmark — Day 7 (No KV Cache, Single Request)

## Setup

| Parameter          | Value                              |
|--------------------|------------------------------------|
| Model              | GPT-2 124M (custom implementation) |
| Parameters         | 124,439,808 (160 state_dict keys)  |
| Precision          | fp32                               |
| GPU                | NVIDIA A100-SXM4-80GB              |
| Peak TFLOPS (fp32) | 19.5                               |
| PyTorch            | 2.4.0+cu121                        |
| Python             | 3.10.18                            |
| KV Cache           | None                               |
| Batching           | Single request                     |
| Sampling           | Greedy (argmax)                    |

---

## 1. Latency Benchmark

**Goal:** Measure how prompt length affects end-to-end latency.

**Config:** Fixed generation = 50 tokens, varying prompt length.

| Prompt Len | Latency (s) | Tok/s | Peak Mem (MB) | GPU Util (%) | Mem Util (%) | MFU (%) |
|------------|-------------|-------|---------------|--------------|--------------|---------|
| 64         | 0.277       | 180.3 | 580           | 73           | 7            | 0.23    |
| 256        | 0.396       | 126.2 | 659           | 96           | 9            | 0.16    |
| 512        | 0.625       | 80.0  | 765           | 98           | 13           | 0.10    |

**Observations:**

- Latency scales with prompt length due to increasing prefill cost (attention is O(n²))
- Throughput drops at 512 tokens — without KV cache, each decode step recomputes attention over the full sequence
- Memory grows linearly with prompt length (~0.4 MB per additional token in activations)
- GPU utilization increases with longer prompts (larger matmuls saturate SMs better)

---

## 2. Throughput Benchmark

**Goal:** Measure sustained token generation rate across different generation lengths.

**Config:** Fixed prompt = `"The meaning of life is"`, varying `max_tokens`.

| Max Tokens | Actual Tokens | Latency (s) | Tok/s | Peak Mem (MB) | GPU Util (%) | Mem Util (%) | MFU (%) |
|------------|---------------|-------------|-------|---------------|--------------|--------------|---------|
| 10         | 10            | 0.075       | 132.7 | 540           | 36           | 4            | 0.17    |
| 50         | 50            | 0.394       | 126.9 | 556           | 33           | 4            | 0.16    |
| 100        | 100           | 0.756       | 132.4 | 576           | 59           | 5            | 0.17    |
| 200        | 200           | 1.182       | 169.2 | 616           | 85           | 8            | 0.22    |
| 300        | 300           | 1.867       | 160.7 | 654           | 96           | 9            | 0.21    |
| 400        | 400           | 2.749       | 145.5 | 697           | 98           | 10           | 0.19    |
| 500        | 500           | 3.819       | 130.9 | 733           | 98           | 10           | 0.17    |
| 600        | 600           | 5.018       | 119.6 | 772           | 99           | 12           | 0.15    |
| 700        | 700           | 6.568       | 106.3 | 810           | 99           | 17           | 0.14    |
| 800        | 800           | 8.395       | 95.3  | 850           | 99           | 20           | 0.12    |
| 900        | 900           | 10.492      | 85.8  | 890           | 99           | 22           | 0.11    |
| 1000       | 1000          | 12.881      | 77.6  | 930           | 99           | 23           | 0.10    |

**Observations:**

- Throughput is roughly flat (~130–170 tok/s) — confirms the model is memory-bandwidth bound
- Peak throughput at 200 tokens (169.2 tok/s) — sweet spot between overhead amortization and quadratic attention cost
- Tok/s degrades beyond 300 tokens — without KV cache, per-step cost grows O(n²) with sequence length
- EOS never fired (Actual = Max) — expected for greedy decoding from a generic prompt
- GPU utilization scales from 36% → 98% as generation length increases

---

## 3. Profiler Breakdown

**Goal:** Identify which operations dominate CUDA time.

**Config:** Prompt = `"The capital of France is"`, 50 tokens, `torch.profiler` with CUDA + shapes.

### Sorted by `cuda_time_total` (top 10)

| Op                                |CUDA Total| Self CUDA % | CUDA Time Avg |# Calls |
|-----------------------------------|----------|-------------|---------------|--------|
| `aten::addmm`                     | 64.84 ms | 53.2%       |    27.02 μs   | 2400   |
| `ampere_sgemm_64x32_sliced1x4_nn` | 37.74 ms | 30.9%       |    33.45 μs   | 1128   |
| `aten::matmul`                    | 24.12 ms | 0.0%        |    19.30 μs   | 1250   |
| `aten::bmm`                       | 12.88 ms | 10.5%       |    10.74 μs   | 1200   |
| `cutlass::Kernel`                 | 11.86 ms | 9.7%        |    19.01 μs   | 624    |
| `aten::mm`                        | 11.86 ms | 9.2%        |    224.81 μs  | 50     |
| `ampere_sgemm_32x32_sliced1x4_nn` | 10.48 ms | 8.6%        |    16.18 μs   | 648    |
| `ampere_sgemm_128x128_nn`         | 8.05 ms  | 6.6%        |    17.67 μs   | 3000   |
| `aten::mul`                       | 6.65 ms  | 5.4%        |    2.22 μs    | 1250   |
| `aten::layer_norm`                | 6.35 ms  | 0.0%        |    5.08 μs    | 2450   |

**Self CPU time total: 367 ms** | **Self CUDA time total: 122 ms**

### Key Findings

- **Matmuls dominate:** `aten::addmm` + `aten::bmm` + `aten::mm` = ~73% of CUDA time
- **Actual CUDA kernels:** `ampere_sgemm_*` and `cutlass::Kernel` are the underlying GPU kernels for the matmuls
- **Wrapper ops have 0% self time:** `aten::matmul` and `aten::layer_norm` dispatch to sub-ops — they show high total time but zero self time
- **Elementwise ops:** `aten::mul` = ~6% combined (GeLU activations, residual connections, attention scaling)

---

## 4. Key Takeaways

### GPU Utilization ≠ Efficiency

- **GPU Utilization: 98%** — the GPU is busy nearly all the time
- **MFU: 0.17%** — but it's only achieving 0.17% of theoretical peak FLOPS
- The GPU is busy **waiting on memory transfers**, not doing useful compute
- This is expected for a small model (124M) with autoregressive decoding (matrix-vector, not matrix-matrix operations)

### What MFU Tells Us

```bash
MFU = (2 × N_params × tokens/sec) / peak_TFLOPS × 100

At 130 tok/s:
MFU = (2 × 124 × 10⁶ × 130) / (19.5 × 10¹²) × 100 = 0.17%
```

### Bottlenecks Identified

1. **No KV cache** → redundant attention recomputation each step → O(n²) per token
2. **No batching** → single sequence, GPU compute units mostly idle
3. **fp32** → half the throughput vs fp16/bf16 on tensor cores
4. **Small model** → each decode step is a matrix-vector multiply, memory-bandwidth bound

### Baseline Reference for Future Optimization

| Metric                            | Baseline Value      |
|-----------------------------------|---------------------|
| Throughput                        | ~130–170 tok/s      |
| MFU                               | ~0.17%              |
| Peak Memory                       | 540–765 MB          |
| GPU Utilization                   | 36–98%              |
| Latency (50 tokens, short prompt) | ~0.4s               |
| CUDA time per token               | ~2.4 ms             |
| Dominant op                       | `aten::addmm` (53%) |

These numbers are the baseline to beat. Expected improvements:

- **KV Cache** (Day 9): Reduces per-step attention from O(n²) to O(n) → significant tok/s improvement for longer sequences
- **Batching** (Day 11): Multiple sequences in parallel → higher MFU and throughput
- **fp16** (future): 2× tensor core throughput → higher MFU
