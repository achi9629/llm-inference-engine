# LLM Inference Engine

Production-inspired LLM inference engine built from scratch in PyTorch, inspired by [vLLM](https://github.com/vllm-project/vllm) architecture. Implements KV caching, continuous batching, paged memory management, and async serving — with benchmarks at each optimization layer.

---

## TL;DR

- **2.22× decode speedup** — KV cache eliminates redundant attention recomputation at 1K tokens
- **118× throughput** — batched inference saturates GPU compute (bs=512 vs bs=1)
- **8.7× memory reduction** — paged KV cache vs contiguous pre-allocation (bs=256)
- **11.1× system throughput** — continuous batching + paged cache under 64 concurrent users
- **122 tests** across 15 modules | GPT-2 124M on A100 80GB PCIe

---

## Motivation

Production LLM serving systems (vLLM, TGI, TensorRT-LLM) are 100K+ line codebases mixing C++, CUDA, and Python. Understanding *why* they make specific design decisions — KV caching, continuous batching, paged memory — is difficult from reading production code alone.

This project builds the inference stack **from scratch**, implementing each optimization incrementally:

1. **Transformer forward pass** — understand the computation graph
2. **KV Cache** — eliminate redundant recomputation (O(n²) → O(n) per step)
3. **Batched inference** — saturate GPU compute (118× throughput)
4. **Continuous batching** — dynamic scheduling to eliminate idle GPU slots
5. **Paged KV cache** — block-level memory management (8.7× memory reduction)
6. **Async serving** — FastAPI server with backpressure and routing
7. **Load testing** — end-to-end benchmarks under concurrent load (11.1× system throughput)

Each layer builds on the previous one. Benchmarks at each checkpoint quantify the impact, creating a complete understanding of *what matters* and *why* in LLM inference.

---

## Problem Statement

This project isolates and benchmarks each inference optimization individually — measuring what matters and by how much.

**Setup:** Custom GPT-2 124M (from-scratch forward pass) · NVIDIA A100 80GB PCIe · fp32 · greedy decoding

---

## Pipeline

```bash
Transformer Forward Pass → KV Cache → Batched Inference → Continuous Batching → Paged KV Cache → Async Serving
```

---

## Results Summary

| Optimization   | Metric                   | Baseline  | Optimized    | Gain      |
|----------------|--------------------------|-----------|--------------|-----------|
| KV Cache       | Decode latency (512 tok) | 0.625s    | 0.286s       | **2.19×** |
| KV Cache       | Self CUDA time           | 121.9 ms  | 65.1 ms      | **−47%**  |
| Batching       | Throughput (bs=512)      | 155 tok/s | 18,346 tok/s | **118×**  |
| Paged KV Cache | VRAM (bs=256)            | 28,183 MB | 3,220 MB     | **8.7×**  |
| Serving        | System throughput (c=64) | 165 tok/s | 1,840 tok/s  | **11.1×** |

![Memory: Standard vs Paged by Batch Size](assets/plots/paged_memory_vs_batch.png)

---

## Key Findings

1. **Batching is the single largest win.** 118× from bs=1 to bs=512 — GPU utilization goes from <5% to saturated. Everything else is secondary.
2. **KV cache gain is sequence-length-dependent.** At 200 tokens: 1.02×. At 1000: 2.22×. The optimization only matters when recomputation cost dominates.
3. **Paged cache only wins above batch ~32.** Below that, contiguous allocation uses less memory (no block metadata overhead). Crossover at ~24–32 sequences.
4. **Python scatter/gather is the paged cache bottleneck.** ~1.4–1.8× throughput overhead vs contiguous — production systems solve this with fused PagedAttention CUDA kernels.
5. **Paged memory stays nearly flat regardless of batch size** (2,681→3,220 MB) — only allocates blocks actually used. Standard grows linearly and OOMs at batch 1024.
6. **Long prompts don't hurt paged serving.** 9.5× throughput at long sequences vs 11.1× at short — vectorized cache updates scale well.
7. **Kernel dispatch changes with KV cache.** Baseline is `sgemm`-dominated (full matrix × matrix); KV cache shifts to `gemv` (matrix × vector) — 47% CUDA time reduction.

---

## Failure Analysis

| What broke                                                          | Root cause                                                                                     | Lesson                                                                    |
|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Paged cache slower than contiguous at small batch                   | Block metadata + Python-level scatter/gather dominates when fragmentation isn't the bottleneck | Measure crossover point; don't assume paging always wins                  |
| Standard cache OOM at batch 1024, paged survives                    | Contiguous pre-allocates `max_seq_len` per sequence; paged allocates on-demand                 | Pre-allocation trades memory for simplicity — only viable at small batch  |
| Load test showed flat standard throughput regardless of concurrency | Requests serialized at `batch_size=1` — no batching in standard path                           | Batching is not optional for serving; sequential decode wastes GPU cycles |

---

## Implementation Highlights

- **Custom GPT-2 124M** — multi-head attention, transformer blocks, autoregressive generation; weight loading from OpenAI checkpoints (160 parameters, shape validation)
- **KV Cache** — pre-allocated tensors `(B, n_heads, max_seq_len, head_dim)` per layer; decode appends one token's K/V per step
- **Paged KV Cache** — block pool `(num_blocks, n_heads, block_size, head_dim)`, free-list allocator, block table for logical→physical mapping, `PagedCacheContext` adapter for drop-in compatibility
- **Continuous Batching** — iteration-level scheduler that evicts completed sequences and fills vacant slots per decode step; `ContinuousKVCache` with `reset_slot()` for reuse
- **Serving Layer** — FastAPI with `asyncio.Semaphore` (503 at capacity), `asyncio.wait_for` (504 on timeout), background generation loop, per-request futures
- **Profiling** — `torch.profiler` with CUDA event timing, GPU utilization via `pynvml`, MFU calculation

### Paged KV Cache vs PagedAttention

|                  | Paged KV Cache (this project)                                          | PagedAttention (vLLM)                                                                        |
|------------------|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| **What**         | Memory management layer — KV entries stored in fixed-size blocks       | Complete attention algorithm — fused CUDA kernel operating directly on non-contiguous blocks |
| **Analogy**      | Virtual memory pages in an OS                                          | Virtual memory + hardware TLB                                                                |
| **Performance**  | Memory savings + ~1.4–1.8× throughput overhead (Python scatter/gather) | Memory savings + zero throughput overhead (fused kernel)                                     |
| **Reference**    | OS virtual memory concepts                                             | [PagedAttention paper](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023)                 |

---

## Detailed Benchmarks

### Throughput: Baseline vs KV Cache

![Throughput Comparison](assets/plots/throughput_comparison.png)

| Generation Length | Baseline    | KV Cache    | Speedup   |
|-------------------|-------------|-------------|-----------|
| 200 tokens        | 169.2 tok/s | 173.1 tok/s | 1.02×     |
| 500 tokens        | 130.9 tok/s | 172.5 tok/s | 1.32×     |
| 1000 tokens       | 77.6 tok/s  | 172.0 tok/s | **2.22×** |

### Latency: Baseline vs KV Cache

![Latency Comparison](assets/plots/latency_comparison.png)

| Prompt Length | Baseline | KV Cache | Speedup   |
|---------------|----------|----------|-----------|
| 64 tokens     | 0.277s   | 0.277s   | 1.00×     |
| 256 tokens    | 0.396s   | 0.279s   | 1.42×     |
| 512 tokens    | 0.625s   | 0.286s   | **2.19×** |

### GPU Profiler

| Metric          | Baseline                | KV Cache               | Change                  |
|-----------------|-------------------------|------------------------|-------------------------|
| Self CUDA time  | 121.90 ms               | 65.14 ms               | **−46.6%**              |
| Dominant kernel | `sgemm` (matrix-matrix) | `gemv` (matrix-vector) | Kernel dispatch changed |

### Batch Inference Throughput (KV Cache enabled)

![Batch Throughput](assets/plots/batch_throughput.png)

| Batch Size | Tok/s        | Speedup vs bs=1 | Peak Memory |
|------------|--------------|-----------------|-------------|
| 1          | 155 tok/s    | 1×              | 643 MB      |
| 8          | 1,138 tok/s  | 7.3×            | 1,399 MB    |
| 128        | 11,368 tok/s | 73.3×           | 14,359 MB   |
| 512        | 18,346 tok/s | **118.3×**      | 55,831 MB   |

### Paged KV Cache: Memory vs Batch Size

![Memory: Standard vs Paged](assets/plots/paged_memory_vs_batch.png)

| Batch Size | Standard Memory | Paged Memory | Winner    | Savings   |
|------------|-----------------|--------------|-----------|-----------|
| 1          | 643 MB          | 2,681 MB     | Standard  | 0.2×      |
| 16         | 2,263 MB        | 2,712 MB     | Standard  | 0.8×      |
| **32**     | **3,991 MB**    | **2,747 MB** | **Paged** | **1.5×**  |
| 64         | 7,447 MB        | 2,814 MB     | Paged     | 2.6×      |
| 256        | 28,183 MB       | 3,220 MB     | Paged     | **8.7×**  |

### Load Test: Serving Throughput Under Concurrent Load

![Load Test Throughput](assets/plots/load_throughput_vs_concurrency.png)

| Prompt (c=64) | Standard (sequential) | Paged (batched) | Speedup   |
|---------------|-----------------------|-----------------|-----------|
| Short         | 165 tok/s             | 1,840 tok/s     | **11.1×** |
| Medium        | 166 tok/s             | 1,749 tok/s     | **10.5×** |
| Long          | 167 tok/s             | 1,582 tok/s     | **9.5×**  |

| Concurrency | Standard p95 (s) | Paged p95 (s) | Improvement |
|:-----------:|:-----------------:|:-------------:|:-----------:|
| 1           | 0.32              | 0.46          | 0.7×        |
| 8           | 2.42              | 0.52          | **4.7×**    |
| 32          | 9.66              | 1.10          | **8.8×**    |
| 64          | 19.37             | 1.73          | **11.2×**   |
| 128         | 36.66             | 3.80          | **9.6×**    |

> Full benchmark details: [baseline](docs/benchmark/baseline_benchmark.md) | [kv_cache](docs/benchmark/kv_cache_benchmark.md) | [batched](docs/benchmark/batched_kv_cache_benchmark.md) | [continuous_batching](docs/benchmark/continuous_batching_benchmark.md) | [paged_cache](docs/benchmark/paged_kv_cache_benchmark.md) | [load_test](docs/benchmark/load_test_benchmark.md)

---

## Project Structure

```bash
src/llm_engine/
├── model/GPT2/           # Custom transformer (attention, block, feedforward)
├── inference/            # Generator, sampler, inference engine
├── cache/                # KV cache, continuous KV cache, paged KV cache, memory allocator, block table
├── scheduler/            # Batch scheduler, continuous batching scheduler, request queue
├── tokenizer/            # HuggingFace tokenizer wrapper
├── serving/              # FastAPI server, request handler, async router, client
├── config/               # YAML config loader (model, scheduler, server)
├── utils/                # Profiler, GPU monitor, weight loader
```

---

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start server
PYTHONPATH=. python scripts/run_server.py

# Send request
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is", "max_tokens": 50}'

# Run benchmarks
PYTHONPATH=. python benchmarks/latency/latency_benchmark.py
PYTHONPATH=. python benchmarks/throughput/throughput_benchmark.py
PYTHONPATH=. python benchmarks/throughput/paged_kv_cache_benchmark.py

# Run tests (122 tests, 15 files)
PYTHONPATH=src python -m pytest tests/ -v
```

---

## Future Extensions

- Mistral 7B support (RoPE, GQA, RMSNorm, SwiGLU)
- Speculative decoding (GPT-2 small drafts, medium verifies)
- Prefix caching (reuse KV blocks across shared system prompts)
- Custom Triton FlashAttention + PagedAttention kernels
- Memory-aware scheduler admission (block budget check before filling slots)

---

## Documentation

| Document                                                 | Description                                                               |
|----------------------------------------------------------|---------------------------------------------------------------------------|
| [architecture.md](docs/concepts/architecture.md)         | System architecture — 6-layer component diagram, data flow, source map    |
| [caching.md](docs/concepts/caching.md)                   | KV cache variants — standard, continuous, paged — with memory comparison  |
| [design_decisions.md](docs/concepts/design_decisions.md) | Key design tradeoffs with rationale and vLLM/TGI comparisons              |
| [paged_kv_cache.md](docs/concepts/paged_kv_cache.md)     | Paged KV cache deep dive — memory allocator, block table, adapter pattern |
| [scheduling.md](docs/concepts/scheduling.md)             | Continuous batching scheduler design                                      |
| [serving_layer.md](docs/concepts/serving_layer.md)       | Serving architecture — HTTP → Router → Scheduler → Engine                 |

---

## Hardware

|                    |                       |
|--------------------|-----------------------|
| GPU                | NVIDIA A100 80GB PCIe |
| Peak TFLOPS (fp32) | 19.5                  |
| PyTorch            | 2.4.0+cu121           |
| Python             | 3.10.18               |

## License

MIT License — see [LICENSE](LICENSE) for details.
