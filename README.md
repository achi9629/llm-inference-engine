# LLM Inference Engine

A lightweight LLM inference engine built from scratch in PyTorch, inspired by [vLLM](https://github.com/vllm-project/vllm) architecture. Implements core inference optimizations used in production serving systems.

## Features (Implemented)

- Custom GPT-2 124M transformer (from-scratch forward pass, no HuggingFace model)
- Autoregressive text generation with greedy decoding
- **KV Cache** — pre-allocated per-layer cache for O(n) decode instead of O(n²)
- Benchmark suite (latency, throughput, GPU profiler)
- Pretrained weight loading from OpenAI GPT-2 checkpoints

## Features (Planned)

- Batch inference with padding + attention masks
- Continuous batching scheduler
- Paged KV cache (block-level memory management)
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

> Full benchmark details: [baseline_benchmark.md](docs/baseline_benchmark.md) | [kv_cache_benchmark.md](docs/kv_cache_benchmark.md)

## Project Structure

```bash

src/llm_engine/
├── model/GPT2/           # Custom transformer (attention, block, feedforward)
├── inference/            # Generator, sampler, inference engine
├── cache/                # KV cache implementation
├── tokenizer/            # HuggingFace tokenizer wrapper
├── utils/                # Profiler, GPU monitor, weight loader

```

## Quick Start

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Benchmarks

```bash
# Latency benchmark
PYTHONPATH=. python benchmarks/latency/latency_benchmark.py

# Throughput benchmark
PYTHONPATH=. python benchmarks/throughput/throughput_benchmark.py

# GPU profiler
PYTHONPATH=. python -B benchmarks/profiler/profiler_benchmark.py
```

### Run Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## Technical Details

- **KV Cache**: Pre-allocated zero tensors per layer, shape `(B, n_heads, max_seq_len, head_dim)`. During decode, only the new token's K/V are appended. Attention computes `Q_new @ K_cached^T` instead of reprocessing the full sequence.
- **Weight Loading**: Maps OpenAI GPT-2 checkpoint keys to custom model architecture (160 parameters, 124M total).
- **Profiling**: `torch.profiler` with CUDA event timing, GPU utilization via `pynvml`, MFU calculation.

## Hardware

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| Peak TFLOPS (fp32) | 19.5 |
| PyTorch | 2.4.0+cu121 |
| Python | 3.10.18 |
