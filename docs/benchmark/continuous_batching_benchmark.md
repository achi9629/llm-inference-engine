# Continuous Batching Benchmark — Day 13 (Static vs Continuous Batching)

## Setup

| Parameter          | Value                                                       |
|--------------------|-------------------------------------------------------------|
| Model              | GPT-2 124M (custom implementation)                          |
| Parameters         | 124,439,808 (160 state_dict keys)                           |
| Precision          | fp32                                                        |
| GPU                | NVIDIA A100-SXM4-80GB                                       |
| Peak TFLOPS (fp32) | 19.5                                                        |
| PyTorch            | 2.4.0+cu121                                                 |
| Python             | 3.10.18                                                     |
| KV Cache           | Pre-allocated per-layer (B, n_heads, max_seq_len, head_dim) |
| Sampling           | Greedy (argmax)                                             |
| Max Tokens         | 50 per sequence                                             |
| Batch Size         | 4                                                           |
| Total Prompts      | 24 (varying lengths)                                        |

---

## 1. Static Batching Results

**Method:** 6 batches of 4 prompts each. Each batch runs `engine.generate()` independently. Next batch starts only after previous batch finishes completely.

| Batch | Total Tokens | Latency (s) | Tok/s  | Peak Mem (MB) | GPU Util (%) | MFU (%) |
|-------|-------------|-------------|--------|---------------|--------------|---------|
| 1     | 200         | 0.342       | 584.6  | 1,399.4       | 0.0          | 0.75    |
| 2     | 200         | 0.341       | 586.9  | 1,399.4       | 112.0        | 0.75    |
| 3     | 200         | 0.341       | 586.1  | 1,399.4       | 112.0        | 0.75    |
| 4     | 200         | 0.342       | 585.4  | 1,399.4       | 112.0        | 0.75    |
| 5     | 200         | 0.341       | 586.8  | 1,399.4       | 112.0        | 0.75    |
| 6     | 200         | 0.341       | 586.2  | 1,399.4       | 112.0        | 0.75    |

**Average: ~586 tok/s per batch, ~2.05s total for 1200 tokens**

---

## 2. Continuous Batching Results

**Method:** All 24 prompts queued upfront. `ContinuousBatchingScheduler.step()` fills 4 slots, generates, marks finished, refills. Runs in a loop until queue is empty.

| Total Tokens | Latency (s) | Tok/s  | Peak Mem (MB) | GPU Util (%) | MFU (%) |
|-------------|-------------|--------|---------------|--------------|---------|
| 1200        | 2.032       | 590.7  | 1,399.4       | 112.0        | 0.75    |

**Total: 590.7 tok/s, 2.032s for 1200 tokens**

---

## 3. Comparison

| Metric                | Static Batching | Continuous Batching | Difference |
|-----------------------|-----------------|---------------------|------------|
| Total Time (1200 tok) | ~2.05s          | 2.032s              | ~0.9% faster |
| Avg Tok/s             | ~586            | 590.7               | ~0.8% faster |
| Peak Memory           | 1,399.4 MB      | 1,399.4 MB          | Same       |
| Scheduler Overhead    | None            | Negligible          | ~0         |

---

## 4. Why Are They Nearly Identical?

The results are **expected to be the same** for GPT-2 124M with greedy decoding. Here's why:

### No Early EOS

GPT-2 124M with greedy decoding almost never generates EOS before hitting `max_tokens=50`. Every request runs for exactly 50 tokens. This means:

- **No requests finish early** — all 4 requests in a batch finish at the same step
- **No empty slots to fill** — continuous batching has nothing to replace
- **Both strategies process 6 batches of 4** — identical workload

### When Continuous Batching Shows Real Benefit

Continuous batching outperforms static batching when:

| Scenario                        | Why It Helps                                      |
|---------------------------------|---------------------------------------------------|
| Variable-length outputs         | Short outputs finish early, slots get reused       |
| Models that generate EOS        | Larger models (LLaMA 7B+) produce real EOS tokens  |
| High request volume             | Queue refills slots instantly, GPU stays saturated |
| Mixed prompt lengths            | Short prompts finish first, long prompts keep going |

### Expected Impact at Scale

```
Static:   [A, B, C, D] -> A finishes at step 10, BCD run to step 50
          Slots wasted: 40 steps x 1 slot = 40 slot-steps wasted

Continuous: [A, B, C, D] -> A finishes at step 10, E fills immediately
            Slots wasted: 0
```

**With a model that produces variable-length outputs, continuous batching can improve throughput by 2-3x** by eliminating idle GPU slots.

---

## 5. Key Takeaway

> The scheduler adds **zero measurable overhead** (~0.8% difference is within noise). The infrastructure is in place — the throughput benefit will appear naturally when sequences finish at different times (Day 18 load testing with concurrent users, or with larger models).

---

## Benchmark Progression

| Day | Feature                   | Tok/s (best)    | Improvement            |
|-----|---------------------------|-----------------|------------------------|
| 7   | Baseline (no cache)       | 169 tok/s       | —                      |
| 9   | + KV Cache                | 174 tok/s       | 1.03x (single request) |
| 11  | + Batching (bs=512)       | 18,346 tok/s    | 118x (from bs=1)       |
| 13  | + Continuous Batching     | 591 tok/s (bs=4)| Same as static (expected) |

**Next:** Day 15-16 Paged KV Cache (memory-efficient block allocation)
