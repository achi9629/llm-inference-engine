# Throughput vs Latency in LLM Inference

## Why Both Matter

For a **single user**, throughput and latency are interchangeable:

$$\text{throughput} = \frac{\text{tokens generated}}{\text{latency}}$$

If one request generates 50 tokens in 0.5s → throughput = 100 tok/s. One metric gives you the other.

For **multiple concurrent users**, they decouple — and often move in opposite directions.

---

## The Tradeoff

| Scenario                   | Per-User Latency | System Throughput |
|----------------------------|------------------|-------------------|
| 1 user, 50 tokens          | 0.5s             | 100 tok/s         |
| 8 users, batched together  | 1.0s             | 400 tok/s         |
| 32 users, batched together | 2.5s             | 640 tok/s         |

**Batching** lets the GPU process multiple sequences in a single forward pass. This is more compute-efficient (better GPU utilization, higher total tok/s), but each individual user waits longer because:

1. **Shared compute** — GPU time is split across all sequences in the batch
2. **Padding waste** — sequences of different lengths get padded to the longest, burning cycles on pad tokens
3. **Memory pressure** — larger batches consume more KV cache memory, potentially causing evictions or slower allocation

> More users → higher throughput, worse latency per user.

---

## What Each Metric Tells You

| Metric         | Perspective   | Question It Answers                                                   |
|----------------|---------------|-----------------------------------------------------------------------|
| **Latency**    | User-facing   | "How long do I wait for my response?"                                 |
| **Throughput** | System-facing | "How many tokens can the system produce per second across all users?" |

A system with **high throughput but high latency** serves many users but each one waits a long time (batch-heavy strategy).

A system with **low latency but low throughput** gives fast responses but can only handle a few users (no batching).

**Production systems optimize for both**: maximize throughput while keeping latency below an SLA (e.g., p99 < 2s).

---

## Why Reporting Only One Is Misleading

### Reporting only throughput

- "Our system does 5000 tok/s!" — but with 100 concurrent users, each waits 10 seconds. That's unusable for interactive chat.

### Reporting only latency

- "Each request completes in 200ms!" — but if you're processing one at a time, total throughput is 250 tok/s. That serves maybe 2-3 users.

### What to report

For a complete picture under load:

- **Throughput**: total tok/s at each concurrency level
- **Latency distribution**: p50, p90, p95, p99 — not just mean (mean hides tail latency)
- **Concurrency level**: how many simultaneous users

---

## In This Project

| Benchmark              | What It Measures                 | Metrics                                          |
|--------------------- --|----------------------------------|--------------------------------------------------|
| Day 7/9 Latency        | Single user, no concurrency      | Latency ≈ 1/throughput (interchangeable)         |
| Day 18 Load Stress     | Batch size scaling, OOM boundary | Peak memory per batch size (not latency-focused) |
| Day 18 Task 4          | Single-user warm vs cold latency | Latency floor (best case)                        |
| Day 19 Concurrent Load | 4-32 simultaneous users          | Both: throughput + p50/p90/p95/p99 latency       |

The single-user latency test (Task 4) establishes the **floor** — the best latency any user can possibly experience. The concurrent load test (Day 19) shows how that floor degrades as the system is loaded, and what throughput gain you get in exchange.

---

## The Batching Equation

For a system with batch size $B$ and single-request latency $L_1$:

- **Ideal throughput scaling**: $\text{throughput} = B \times \frac{\text{tokens}}{L_1}$ (if batching were free)
- **Actual**: batching has overhead, so throughput scales sub-linearly
- **Latency**: $L_B \geq L_1$ always — batching never helps individual latency

The ratio $\frac{L_B}{L_1}$ is the **latency penalty** of batching. The ratio $\frac{\text{throughput}_B}{B \times \text{throughput}_1}$ is the **batching efficiency**.

Both numbers together tell you whether batching is worth it at a given concurrency level.
