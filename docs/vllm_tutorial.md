# vLLM — Internals & Usage Guide

## What is vLLM?

vLLM is a high-throughput LLM serving engine. Its key innovation is **PagedAttention** — a memory management technique that eliminates KV cache fragmentation by storing KV cache in non-contiguous memory blocks (like OS virtual memory paging).

---

## Core Concepts (Interview-Ready)

### 1. PagedAttention

- Traditional KV cache: pre-allocates contiguous memory per sequence → wastes memory on padding, can't share across requests
- PagedAttention: KV cache stored in fixed-size **blocks** (like memory pages), mapped via a **block table**
- Enables near-zero waste, dynamic allocation, and memory sharing (e.g., beam search shares prefix blocks via copy-on-write)

### 2. Continuous Batching

- Static batching: wait for all requests in a batch to finish before processing new ones
- Continuous batching: as soon as one request finishes, a new request takes its slot immediately
- Result: GPU is never idle waiting for the longest sequence in a batch

### 3. Scheduler

- Manages request queue (waiting → running → swapped)
- Decides which requests get GPU blocks each iteration
- Preemption policy: when memory is tight, swap out lower-priority sequences to CPU

### 4. KV Cache Memory Management

- **Block Manager**: allocates/frees physical blocks, tracks free block pool
- **Block Table**: maps logical block index → physical block index per sequence
- **Swapping**: moves KV cache blocks between GPU ↔ CPU when memory pressure hits
- **Prefix Caching**: reuses KV cache blocks for shared prompt prefixes

### 5. Tensor Parallelism

- Splits model across multiple GPUs (column/row parallel for linear layers)
- All-reduce for synchronization
- Enables serving models larger than single GPU memory

---

## Architecture Overview

```bash
Client Request
      │
      ▼
┌─────────────┐
│  API Server │  (FastAPI / OpenAI-compatible endpoint)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LLM Engine │  (orchestrates scheduling + execution)
└──────┬──────┘
       │
       ├──► Scheduler (decides which seqs run this step)
       │
       ├──► Block Manager (allocates KV cache blocks)
       │
       └──► Model Executor
              │
              ├──► Worker (per GPU)
              │      ├── Model Runner (forward pass)
              │      └── Cache Engine (KV block ops)
              │
              └──► (Optional) Ray for distributed)
```

---

## Python API Usage

### Offline Inference (Batch)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="mistralai/Mistral-7B-v0.3",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,    # fraction of GPU mem for KV cache
    max_model_len=4096,            # max sequence length
    tensor_parallel_size=1,        # number of GPUs
)

sampling_params = SamplingParams(
    temperature=0,          # greedy decoding
    max_tokens=50,
    ignore_eos=True,        # force full generation
    top_p=1.0,
)

prompts = ["Explain PagedAttention in one sentence."]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
    print(f"Tokens generated: {len(output.outputs[0].token_ids)}")
```

### Online Serving (OpenAI-compatible API)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-v0.3 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --port 8000
```

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.3",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=50,
)
```

### Benchmarking Latency

```python
from time import perf_counter
import torch

torch.cuda.reset_peak_memory_stats(0)
torch.cuda.synchronize()
start = perf_counter()

outputs = llm.generate([prompt], sampling_params)

torch.cuda.synchronize()
end = perf_counter()

latency = end - start
num_tokens = len(outputs[0].outputs[0].token_ids)
tokens_per_sec = num_tokens / latency
peak_mem_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
```

---

## Key Constructor Parameters

| Parameter | What it does |
|-----------|-------------|
| `model` | HuggingFace model name or local path |
| `dtype` | `"auto"`, `"bfloat16"`, `"float16"`, `"float32"` |
| `gpu_memory_utilization` | Fraction of GPU mem for KV cache (0.0–1.0) |
| `max_model_len` | Max context length (tokens) |
| `tensor_parallel_size` | Number of GPUs for tensor parallelism |
| `enforce_eager` | Disable CUDA graphs (useful for debugging) |
| `trust_remote_code` | Allow custom model code from HF |
| `quantization` | `"awq"`, `"gptq"`, `"squeezellm"`, etc. |
| `swap_space` | CPU swap space in GB for KV cache offloading |

---

## SamplingParams

| Parameter | Purpose |
|-----------|---------|
| `temperature` | 0 = greedy, higher = more random |
| `top_p` | Nucleus sampling threshold |
| `top_k` | Top-k sampling |
| `max_tokens` | Max tokens to generate |
| `ignore_eos` | Don't stop at EOS (for benchmarking) |
| `presence_penalty` | Penalize repeated tokens |
| `frequency_penalty` | Penalize frequent tokens |
| `stop` | Stop strings |
| `n` | Number of completions per prompt |
| `best_of` | Generate N, return best (beam-like) |

---

## How vLLM Achieves High Throughput

| Technique | Improvement |
|-----------|-------------|
| PagedAttention | Eliminates KV cache fragmentation → fits more sequences in memory |
| Continuous Batching | No wasted GPU cycles waiting for longest seq |
| CUDA Graphs | Reduces kernel launch overhead for decode steps |
| FlashAttention | Fused attention kernel, reduced memory bandwidth |
| Prefix Caching | Shared system prompts don't re-compute KV |
| Speculative Decoding | Draft model proposes tokens, main model verifies in parallel |

---

## Interview Questions & Answers

**Q: Why is vLLM faster than naive HuggingFace generate()?**
> HF allocates contiguous KV cache per sequence (wasteful), processes one batch at a time (no continuous batching), and doesn't use CUDA graphs. vLLM's PagedAttention + continuous batching + CUDA graphs = 2-4x higher throughput.

**Q: What's the difference between PagedAttention and FlashAttention?**
> FlashAttention is a *kernel optimization* — fuses attention computation to reduce memory bandwidth. PagedAttention is a *memory management* strategy — stores KV cache in non-contiguous blocks. They're complementary; vLLM uses both.

**Q: When would you NOT use vLLM?**
> - Models not supported (custom architectures without HF integration)
> - Single-request latency-critical apps (vLLM's overhead isn't worth it for 1 request)
> - Training (vLLM is inference-only)
> - Edge deployment (too heavy, use llama.cpp or ONNX Runtime)

**Q: How does vLLM handle memory pressure?**
> The scheduler preempts running sequences — either **swap** (move KV blocks to CPU) or **recompute** (discard and recompute later). Priority is typically FCFS.

**Q: What is `gpu_memory_utilization`?**
> Fraction of free GPU memory (after model loading) allocated for KV cache blocks. Higher = more concurrent sequences but risk of OOM during spikes.

**Q: How does continuous batching work in vLLM?**
> Each decode step, the scheduler picks which sequences to run. Finished sequences are immediately replaced by waiting ones. The batch composition can change every single iteration.

**Q: Explain the block table.**
> Each sequence has a logical → physical block mapping. Logical block 0 maps to some physical block N in GPU memory. This indirection enables: non-contiguous storage, sharing (copy-on-write for beam search), and dynamic growth.

---

## Comparison: Your Engine vs vLLM

| Feature         | Your Engine                  | vLLM                        |
|-----------------|------------------------------|-----------------------------|
| KV Cache        | Contiguous or Paged (custom) | PagedAttention              |
| Batching        | Static / Continuous          | Continuous                  |
| Scheduling      | Custom FCFS                  | Priority + preemption       |
| Serving         | FastAPI                      | FastAPI (OpenAI-compatible) |
| CUDA Graphs     | No                           | Yes                         |
| FlashAttention  | No                           | Yes                         |
| Tensor Parallel | No                           | Yes                         |
| Quantization    | No                           | AWQ, GPTQ, SqueezeLLM       |

This comparison is what makes your project valuable — you built the fundamentals from scratch and understand WHY each optimization matters because you measured the gap.

---

## Resume Line

```bash
Tools: vLLM, PyTorch, CUDA, FastAPI, HuggingFace Transformers
```

**How to defend:**

- "I benchmarked my custom inference engine against vLLM to quantify the impact of PagedAttention and continuous batching"
- "I used vLLM's offline API for latency profiling across prompt lengths on Mistral-7B"
- "I understand vLLM internals — PagedAttention, block manager, scheduler preemption, CUDA graphs — because I implemented simplified versions in my engine"
