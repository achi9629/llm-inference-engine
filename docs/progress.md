# LLM Inference Engine - 3-Week Implementation Checklist

## Status Legend

| Status | Emoji |
| ----------- | ----- |
| Done | ✅ |
| Not done | ⬜ |
| In progress | 🔄 |
| Blocked | 🚫 |

## Week 1: Core Foundations (Mar 9-15)

### Day 1 - Project Setup + Tokenizer (Mar 9)

- ✅ Fill `requirements.txt` (torch, transformers, sentencepiece, pyyaml, fastapi, uvicorn, numpy)
- ✅ Fill `pyproject.toml` (package metadata, dependencies)
- ✅ Fill `configs/model_config.yaml` (model_name, max_seq_len, device, dtype)
- ✅ Fill `configs/scheduler_config.yaml` (max_batch_size, max_tokens_per_batch, scheduling_policy)
- ✅ Fill `configs/server_config.yaml` (host, port, timeout, max_concurrent_requests)
- ✅ Implement `src/llm_engine/tokenizer/tokenizer.py` (HuggingFace tokenizer wrapper: encode, decode)
- ✅ Verify tokenizer with sample text

### Day 2 - Basic Generation Loop (Mar 10)

- ✅ Implement `src/llm_engine/inference/sampler.py` (greedy decoding via argmax)
- ✅ Implement `src/llm_engine/inference/generator.py` (autoregressive loop: prefill -> decode -> stop at EOS/max_len)
- ✅ Implement `src/llm_engine/inference/inference_engine.py` (orchestrator: tokenize -> generate -> detokenize)

### Day 3 - Model Integration I (Mar 11)

- ✅ Integrate custom Transformer decoder forward pass
- ✅ Run a smoke test on a tiny prompt through the full inference path

### Day 4 - Model Integration II (Mar 12)

- ✅ Load pretrained weights into custom Transformer
- ✅ Add clear logging for loaded/missing/unexpected parameter groups

### Day 5 - Validation Day (Mar 13)

- ✅ Validate weight mapping (keys/shapes, missing/unexpected params report)
- ✅ Generate tokens until EOS or max token limit
- ✅ Verify correctness with several test prompts

### Day 6 - Profiling Tooling (Mar 14)

- ✅ Implement `src/llm_engine/utils/profiler.py` (torch.profiler wrapper)
- ✅ Implement `src/llm_engine/utils/gpu_monitor.py` (GPU utilization tracking)
- ✅ Implement `benchmarks/metrics.py` (tokens/sec, latency, MFU, GPU util, memory usage collectors)

### Day 7 - Baseline Benchmark (Mar 15)

- ✅ Measure latency for prompt lengths: 64, 256, 512 tokens
- ✅ Measure tokens/sec throughput (baseline, no cache)
- ✅ Inspect GPU utilization with torch.profiler
- ✅ Document baseline performance numbers

---

## Week 2: Caching + Scheduling (Mar 16-22)

### Day 8 - KV Cache Core (Mar 16)

- ✅ Design KV cache tensor structure per transformer layer
- ✅ Implement `src/llm_engine/cache/kv_cache.py` (pre-allocated K/V tensors per layer)

### Day 9 - KV Cache Integration (Mar 17)

- ✅ Implement `src/llm_engine/model/attention.py` (modified attention to reuse cached K/V)
- ✅ Update cache during each decoding step
- ✅ Benchmark speed improvement vs no-cache baseline
- ✅ Record KV cache speedup numbers

### Day 10 - Batch Inference Basics (Mar 18)

- ✅ Extend `src/llm_engine/inference/generator.py` for batched inputs
- ✅ Handle different prompt lengths within a batch (padding/attention masks)

### Day 11 - Batch Inference Evaluation (Mar 19)

- ✅ Run batch inference with 2-8 prompts simultaneously
- ✅ Measure throughput improvements vs single-request
- ✅ Record batch inference throughput numbers

### Day 12 - Request Queue + Scheduler (Mar 20)

- ✅ Implement `src/llm_engine/scheduler/request.py` (request dataclass: prompt, token_ids, state, timestamps)
- ✅ Implement `src/llm_engine/scheduler/request_queue.py` (FIFO queue for pending requests)
- ✅ Implement `src/llm_engine/scheduler/batch_scheduler.py` (group requests into batches respecting memory budget)
- ✅ Simulate asynchronous requests arriving over time

### Day 13 - Continuous Batching + Unit Tests (Mar 21)

- ✅ Implement `src/llm_engine/scheduler/continuous_batching.py` (add new requests mid-batch, evict finished ones)
- ✅ Maintain KV cache separately for each sequence
- ✅ Measure GPU utilization improvement vs static batching
- ✅ Write `tests/test_attention.py`
- ✅ Write `tests/test_kv_cache.py`
- ✅ Write `tests/test_generator.py`
- ✅ Write `tests/test_scheduler.py`
- ✅ Write test cases for continuous batching in the `tests/test_scheduler.py`
- ✅ Write `tests/test_continuous_kv_cache.py`

### Day 14 - Test Stabilization (Mar 22)

- ✅ All tests passing
- ✅ Fix failing edge cases discovered in Week 2 integrations

---

## Week 3: Paged Memory, Serving, Benchmarks, Docs (Mar 23-29)

### Day 15 - Paged KV Cache: Memory Allocator (Mar 23)

- ✅ Implement `src/llm_engine/cache/memory_allocator.py` (block-level GPU memory allocator, free list, fixed-size blocks)
- ✅ Unit test allocator: allocate, free, reuse blocks

### Day 16 - Paged KV Cache: Block Table + Integration (Mar 24)

- ✅ Implement `src/llm_engine/cache/block_table.py` (per-request logical -> physical block mapping)
- ✅ Implement `src/llm_engine/cache/paged_kv_cache.py` (paged attention cache using allocator + block table)
- ✅ Write `tests/test_block_table.py`
- ✅ Write `tests/test_paged_kv_cache.py`
- ✅ Integrate paged cache with continuous batching scheduler
- ✅ Reuse freed blocks when sequences complete
- ✅ Forward pass integration (PagedCacheContext adapter)
- ✅ Implement `src/llm_engine/cache/paged_cache_context.py`
- ✅ Write `tests/test_paged_cache_context.py`
- ✅ Integrate PagedCacheContext with generator.py (pass adapter as kv_cache argument)
- ✅ Integrate PagedCacheContext with inference_engine.py (create adapter from paged cache + block table)
- ✅ Measure GPU memory usage under concurrent requests

### Day 17 - Serving Layer (Mar 25)

- ✅ Implement `src/llm_engine/serving/request_handler.py` (parse HTTP request -> create Request object)
- ✅ Write `tests/test_request_handler.py`
- ✅ Implement `src/llm_engine/serving/router.py` (route to scheduler, return streaming response)
- ✅ Write `tests/test_router.py`
- ✅ Implement `src/llm_engine/serving/api_server.py` (FastAPI server with `/generate` endpoint)
- ✅ Write `tests/test_api_server.py`
- ✅ Implement `src/llm_engine/serving/client.py` (Python client for testing)
- ✅ Implement `examples/server_example.py` (client usage example)
- ✅ Implement `scripts/run_server.py` (one-command server startup)
- ✅ Test end-to-end: client -> server -> generation -> streaming response

### Day 18 - Load Testing (Mar 26)

- ✅ Make Router handle concurrent requests (async or thread-based batching)
- ✅ Add async unit tests in `tests/test_router.py`
- ✅ Paged vs Standard KV cache under load (increase concurrent sequences until OOM)
- ✅ Single-user baseline latency (warm-up vs cold-start)
- ✅ Document results + plots

### Day 19 - Full Benchmark Suite + Utils (Mar 27)

- ✅ Implement `benchmarks/load/load_test.py` (simulate 1-128 concurrent users)
- ✅ Implement `benchmarks/throughput/throughput_test.py`
- ✅ Implement `benchmarks/latency/latency_test.py`
- ✅ Concurrent load test (4, 8, 16, 32 users) — throughput + latency distribution
- ✅ Queue depth / backpressure test (100 requests, max_batch=4)
- ✅ Request arrival patterns (burst vs steady vs Poisson)
- ✅ Test with short, medium, and long prompts
- ✅ Measure p50, p90, p95, p99 latency under load
- ✅ Measure tokens/sec throughput under load
- ✅ Record GPU utilization and memory consumption
- ✅ Document results + plots

### Day 20 - Refinement (Mar 28)

- ✅ Implement `scripts/run_benchmark.py` (run all benchmarks end-to-end)
- ✅ Collect final benchmark numbers across all configurations
- ✅ Write `tests/test_inference.py`
- ✅ Fix edge cases and bugs
- ✅ Optimize batching logic
- ✅ Code cleanup and consistency pass

### Day 21 - Documentation + Portfolio (Mar 29)

- ✅ Write `docs/concepts/architecture.md` (system architecture + diagrams)
- ✅ Write `docs/concepts/design_decisions.md` (key design choices and tradeoffs)
- ✅ Write `docs/concepts/kv_cache.md` (KV cache mechanism explanation)
- ✅ Update `README.md` (overview, architecture, features, benchmarks, how to run, future work)
- ✅ Include benchmark results and performance charts in docs
- ✅ Final GitHub-ready cleanup

## Week 4: LLaMA — Model Architecture (Apr 7-13)

### Day 22 - RMSNorm + RoPE (Apr 7)

- ✅ Implement `src/llm_engine/model/LLaMA/rmsnorm.py` (RMSNorm: no mean subtraction, no bias, float32 upcast)
- ✅ Implement `src/llm_engine/model/LLaMA/rope.py` (Rotary Position Embeddings: precompute freqs, apply rotation to Q/K)

### Day 23 - SwiGLU + GQA Attention (Apr 8)

- ⬜ Implement `src/llm_engine/model/LLaMA/feedforward.py` (SwiGLU: gate_proj, up_proj, down_proj, no bias)
- ⬜ Implement `src/llm_engine/model/LLaMA/attention.py` (Grouped Query Attention: n_kv_heads < n_heads, RoPE on Q/K, repeat_kv)

### Day 24 - LLaMA Block + Transformer (Apr 9)

- ⬜ Implement `src/llm_engine/model/LLaMA/block.py` (pre-norm with RMSNorm + GQA + SwiGLU)
- ⬜ Implement `src/llm_engine/model/LLaMA/transformer.py` (embedding, N blocks, final RMSNorm, untied lm_head)

### Day 25 - Config + Weight Loader (Apr 10)

- ⬜ Create `assets/models/llama/7B/config.json` (vocab_size, n_layers, n_heads, n_kv_heads, d_model, d_ff, rope_theta, etc.)
- ⬜ Create `assets/models/llama/7B/manifest.json` (model_family, weights_file, tokenizer type)
- ⬜ Update `src/llm_engine/utils/weight_loader.py` (map HuggingFace LLaMA checkpoint keys to our module names)

### Day 26 - Tokenizer + Model Dispatch (Apr 11)

- ⬜ Update `src/llm_engine/tokenizer/tokenizer.py` (support SentencePiece / LlamaTokenizer based on manifest tokenizer type)
- ⬜ Update `src/llm_engine/model/__init__.py` (add LLaMA to `load_model()` dispatch)
- ⬜ Update `src/llm_engine/inference/inference_engine.py` (use `n_kv_heads` for KV cache dimensions when GQA)

### Day 27 - Smoke Test + Weight Debugging (Apr 12)

- ⬜ Load LLaMA 7B weights, run forward pass
- ⬜ Compare logits against HuggingFace `LlamaForCausalLM` reference
- ⬜ Fix weight mapping mismatches

### Day 28 - Generation Validation (Apr 13)

- ⬜ Validate end-to-end text generation (coherent output)
- ⬜ Test with paged KV cache (GQA dimensions: n_kv_heads, not n_heads)
- ⬜ Write LLaMA-specific unit tests

## Week 5: LLaMA — Benchmarks + Documentation (Apr 14-20)

### Day 29 - LLaMA Benchmarks (Apr 14)

- ⬜ Latency benchmark (prompt lengths: 64, 256, 512)
- ⬜ Throughput benchmark (single request, batched)
- ⬜ Memory benchmark (standard vs paged KV cache with GQA)

### Day 30 - LLaMA Load Test (Apr 15)

- ⬜ Concurrent load test (1–64 users, paged cache)
- ⬜ Compare LLaMA vs GPT-2 serving throughput
- ⬜ Document load test results

### Day 31 - Documentation + Cleanup (Apr 16)

- ⬜ Update `docs/concepts/architecture.md` (add LLaMA to component diagram)
- ⬜ Write LLaMA concept docs (GQA, RoPE, SwiGLU)
- ⬜ Update `README.md` (add LLaMA to features, benchmark results)
- ⬜ Update `docs/progress.md` — mark LLaMA extension complete

### Day 32-35 - Next Extension (Apr 17-20)

- ⬜ Start next extension (Speculative Decoding or Triton FlashAttention)

---

## Benchmark Checkpoints

| Checkpoint | Day | Metrics | Report |
| ---------- | --- | ------- | ------ |
| Baseline (no cache, single request) | 7 | latency, tokens/sec, GPU util | [baseline.md](benchmark/baseline_benchmark.md) |
| + KV Cache | 9 | same prompts, show speedup | [kv_cache.md](benchmark/kv_cache_benchmark.md) |
| + Batch Inference | 11 | 2-8 prompts, throughput gain | [batched.md](benchmark/batched_kv_cache_benchmark.md) |
| + Continuous Batching | 13 | dynamic join/leave, GPU util improvement | [continuous_batching.md](benchmark/continuous_batching_benchmark.md) |
| + Paged KV Cache | 16 | memory usage under concurrent requests | [paged_kv_cache.md](benchmark/paged_kv_cache_benchmark.md) |
| Full Load Test | 19 | 1-128 users, p95 latency, tokens/sec, GPU memory | [load_test.md](benchmark/load_test_benchmark.md) |

---

## Final Deliverables

- ✅ Working end-to-end inference server
- ✅ KV cache with paged memory management
- ✅ Continuous batching scheduler
- ✅ Benchmark suite with performance progression
- ✅ Complete documentation with architecture diagrams
- ✅ Clean, well-structured codebase on GitHub

## Future Extensions (Post-Project)

- ⬜ Scheduler-driven decode loop (token-level eviction/fill using ContinuousBatchingScheduler + ContinuousKVCache, needed because chat models produce variable-length responses with natural EOS)
- ⬜ Speculative decoding (GPT-2 small drafts, GPT-2 medium verifies)
- ⬜ Prefix caching (reuse KV blocks across requests sharing system prompt)
- ⬜ Benchmark against vLLM on matched workloads
- ⬜ Custom Triton FlashAttention kernel (fused QK^T → softmax → V, tiled, online softmax, O(N) memory)
- ⬜ Custom Triton PagedAttention kernel (read directly from scattered KV blocks — eliminates gather step)
- ⬜ Memory-aware scheduler admission (block budget check before filling slots — needed when KV cache per sequence is large enough to cause MemoryError)
- ⬜ Priority scheduling (swap FIFO deque for heapq in RequestQueue)
- ⬜ Roofline analysis doc (compute vs memory-bandwidth bound at each optimization stage)
- ⬜ GitHub Actions CI workflow (mark GPU tests `@pytest.mark.gpu`, run non-GPU tests in CI)
- ⬜ Guided decoding / structured output (JSON schema-constrained sampling via logit masking)
- ⬜ Add Mistral model support
