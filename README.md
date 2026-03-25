# LLM Inference Engine

A lightweight LLM inference engine built from scratch in PyTorch, inspired by [vLLM](https://github.com/vllm-project/vllm) architecture. Implements core inference optimizations used in production serving systems.

## Motivation

Production LLM serving systems (vLLM, TGI, TensorRT-LLM) are complex codebases with 100K+ lines of C++/CUDA/Python. Understanding *why* they make specific design decisions — KV caching, continuous batching, paged memory — is difficult from reading production code alone.

This project builds an LLM inference engine **from scratch**, implementing each optimization incrementally with benchmarks at every stage. The goal is to deeply understand the inference stack by building it layer by layer:

1. **Transformer forward pass** — understand the computation graph
2. **KV Cache** — eliminate redundant attention recomputation (O(n²) → O(n) per step)
3. **Batched inference** — saturate GPU compute with multiple sequences (118x throughput gain)
4. **Continuous batching** — dynamic scheduling to eliminate idle GPU slots
5. **Paged KV cache** — block-level memory management to eliminate internal fragmentation (8.7x memory reduction at scale)
6. **Serving layer** — FastAPI server with request handling and routing
7. **Load testing** — end-to-end benchmarks under concurrent load

Each layer builds on the previous one. Benchmarks at each checkpoint quantify the impact of each optimization, creating a complete understanding of *what matters* and *why* in LLM inference.

### Paged KV Cache vs PagedAttention

| | Paged KV Cache | PagedAttention |
|---|---|---|
| **What** | Memory management layer — stores KV entries in fixed-size blocks instead of contiguous tensors | Complete attention algorithm — computes Q×K^T, softmax, weighted sum directly on non-contiguous blocks in a single fused CUDA kernel |
| **Components** | Memory allocator, block table, paged KV cache tensor pool | Paged KV cache + custom CUDA attention kernel (`paged_attention_v1/v2`) |
| **Analogy** | Virtual memory pages in an OS | Virtual memory + hardware TLB that translates addresses in-line |
| **This project** | ✅ Fully implemented (Python) | ❌ Not implemented — would require custom CUDA kernels |
| **Performance** | Memory savings (up to 8.7x at large batch sizes). ~2x throughput overhead from Python scatter/gather | Memory savings + zero throughput overhead (fused kernel eliminates Python loops) |
| **Reference** | OS virtual memory concepts | [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023) |

**In short:** Paged KV Cache is the data structure. PagedAttention is the data structure + a fused kernel that operates on it. This project implements the former to understand the memory management principles. Production systems (vLLM) add the latter for zero-overhead paged attention.

## Features (Implemented)

- Custom GPT-2 124M transformer (from-scratch forward pass, no HuggingFace model)
- Autoregressive text generation with greedy decoding
- **KV Cache** — pre-allocated per-layer cache for O(n) decode instead of O(n²)
- Benchmark suite (latency, throughput, GPU profiler)
- Pretrained weight loading from OpenAI GPT-2 checkpoints
- **Batch Inference** — static batching with left padding, attention masks, per-sequence EOS tracking (up to 118x throughput gain)
- **Continuous batching scheduler** — iteration-level scheduling with per-sequence KV cache tracking, dynamic slot eviction and refill (step-based scheduler)
- **Paged KV cache** — block-level memory management with memory allocator, block table, and PagedCacheContext adapter (eliminates internal fragmentation)

## Features (Planned)

- Request queue + priority scheduling
- FastAPI serving layer with token streaming

## Benchmark Results

GPT-2 124M on NVIDIA A100-SXM4-80GB, fp32, single request, greedy decoding.

### Throughput: Baseline vs KV Cache

![Throughput Comparison](assets/plots/throughput_comparison.png)

| Generation Length | Baseline    | KV Cache    | Speedup   |
|-------------------|-------------|-------------|-----------|
| 200 tokens        | 169.2 tok/s | 173.1 tok/s | 1.02x     |
| 500 tokens        | 130.9 tok/s | 172.5 tok/s | 1.32x     |
| 1000 tokens       | 77.6 tok/s  | 172.0 tok/s | **2.22x** |

### Latency: Baseline vs KV Cache

![Latency Comparison](assets/plots/latency_comparison.png)

| Prompt Length | Baseline | KV Cache | Speedup   |
|---------------|----------|----------|-----------|
| 64 tokens     | 0.277s   | 0.277s   | 1.00x     |
| 256 tokens    | 0.396s   | 0.279s   | 1.42x     |
| 512 tokens    | 0.625s   | 0.286s   | **2.19x** |

### GPU Profiler

| Metric         | Baseline                | KV Cache               | Change                  |
|----------------|-------------------------|------------------------|-------------------------|
| Self CUDA time | 121.90 ms               | 65.14 ms               | **-46.6%**              |
| Dominant kernel| `sgemm` (matrix-matrix) | `gemv` (matrix-vector) | Kernel dispatch changed |

### Batch Inference Throughput (KV Cache enabled)

![Batch Throughput](assets/plots/batch_throughput.png)

| Batch Size | Tok/s        | Speedup vs bs=1 | Peak Memory |
|------------|--------------|-----------------|-------------|
| 1          | 155 tok/s    | 1x              | 643 MB      |
| 8          | 1,138 tok/s  | 7.3x            | 1,399 MB    |
| 128        | 11,368 tok/s | 73.3x           | 14,359 MB   |
| 512        | 18,346 tok/s | **118.3x**      | 55,831 MB   |

### Paged KV Cache: Memory vs Batch Size

![Memory: Standard vs Paged by Batch Size](assets/plots/paged_memory_vs_batch.png)

| Batch Size | Standard Memory | Paged Memory | Winner    | Memory Ratio |
|------------|-----------------|--------------|-----------|--------------|
| 1          | 643 MB          | 2,681 MB     | Standard  | 0.2x         |
| 16         | 2,263 MB        | 2,712 MB     | Standard  | 0.8x         |
| **32**     | **3,991 MB**    | **2,747 MB** | **Paged** | **1.5x**     |
| 64         | 7,447 MB        | 2,814 MB     | Paged     | 2.6x         |
| 256        | 28,183 MB       | 3,220 MB     | Paged     | **8.7x**     |

- **Memory crossover at batch_size ~24-32** — below this, standard's pre-allocated cache is smaller; above it, paged's block pool wins
- **Paged memory nearly flat** (2,681→3,220 MB) regardless of batch size — only allocates blocks actually used
- **Throughput tradeoff:** ~2x slower at small batches due to Python-level scatter/gather (no fused CUDA kernel)

> Full benchmark details: [baseline_benchmark.md](docs/benchmark/baseline_benchmark.md) | [kv_cache_benchmark.md](docs/benchmark/kv_cache_benchmark.md) | [batched_kv_cache_benchmark.md](docs/benchmark/batched_kv_cache_benchmark.md) | [continuous_batching_benchmark.md](docs/benchmark/continuous_batching_benchmark.md)| [paged_kv_cache_benchmark.md](docs/benchmark/paged_kv_cache_benchmark.md)

## Project Structure

```bash

src/llm_engine/
├── model/GPT2/           # Custom transformer (attention, block, feedforward)
├── inference/            # Generator, sampler, inference engine
├── cache/                # KV cache, memory allocator, block table, paged kv cache, paged cache context implementation 
├── scheduler/            # Batch scheduler, continuous batching scheduler with paged kv cache and block table integration, 
├── tokenizer/            # HuggingFace tokenizer wrapper
├── serving/              # FastAPI server, request handler, router
├── utils/                # Profiler, GPU monitor, weight loader
```

## Quick Start

### Setup

#### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 2) Install dependencies

````bash
python -m pip install --upgrade pip
pip install -r requirements.txt
````

#### 3) Verify installation

````bash
python -c "import torch, transformers, fastapi; print('OK', torch.__version__, transformers.__version__)"
````

### Run Benchmarks

```bash
# Latency benchmark
PYTHONPATH=. python benchmarks/latency/latency_benchmark.py

# Throughput benchmark
PYTHONPATH=. python benchmarks/throughput/throughput_benchmark.py
PYTHONPATH=. python benchmarks/throughput/continuous_batching_benchmark.py
PYTHONPATH=. python benchmarks/throughput/paged_kv_cache_benchmark.py

# GPU profiler
PYTHONPATH=. python -B benchmarks/profiler/profiler_benchmark.py
```

### Run Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

### Run Example

```bash
PYTHONPATH=. python examples/simple_generation.py
```

## Technical Details

- **KV Cache**: Pre-allocated zero tensors per layer, shape `(B, n_heads, max_seq_len, head_dim)`. During decode, only the new token's K/V are appended. Attention computes `Q_new @ K_cached^T` instead of reprocessing the full sequence.
- **Weight Loading**: Maps OpenAI GPT-2 checkpoint keys to custom model architecture (160 parameters, 124M total).
- **Profiling**: `torch.profiler` with CUDA event timing, GPU utilization via `pynvml`, MFU calculation.
- **Paged KV Cache**: Block pool tensor `(num_blocks, n_layers, 2, block_size, n_heads, head_dim)`. Memory allocator manages a free list of block IDs. Block table maps `(seq_id, block_idx)` to physical blocks. PagedCacheContext adapter wraps these behind `update_cache(layer_idx, k, v)` for drop-in compatibility with the generator.

## Hardware

|                    |                       |
|--------------------|-----------------------|
| GPU                | NVIDIA A100-SXM4-80GB |
| Peak TFLOPS (fp32) | 19.5                  |
| PyTorch            | 2.4.0+cu121           |
| Python             | 3.10.18               |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
