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
- ✅ Implement `src/llm_engine/utils/tensor_utils.py` (padding, batching helpers)
- ✅ Handle different prompt lengths within a batch (padding/attention masks)

### Day 11 - Batch Inference Evaluation (Mar 19)

- ✅ Run batch inference with 2-8 prompts simultaneously
- ✅ Measure throughput improvements vs single-request
- ✅ Record batch inference throughput numbers

### Day 12 - Request Queue + Scheduler (Mar 20)

- ✅ Implement `src/llm_engine/scheduler/request.py` (request dataclass: prompt, token_ids, state, timestamps)
- ✅ Implement `src/llm_engine/scheduler/request_queue.py` (priority queue for pending requests)
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

- ⬜ Implement `src/llm_engine/cache/memory_allocator.py` (block-level GPU memory allocator, free list, fixed-size blocks)
- ⬜ Unit test allocator: allocate, free, reuse blocks

### Day 16 - Paged KV Cache: Block Table + Integration (Mar 24)

- ⬜ Implement `src/llm_engine/cache/block_table.py` (per-request logical -> physical block mapping)
- ⬜ Implement `src/llm_engine/cache/paged_kv_cache.py` (paged attention cache using allocator + block table)
- ⬜ Integrate paged cache with continuous batching scheduler
- ⬜ Write `tests/test_paged_kv_cache.py`
- ⬜ Measure GPU memory usage under concurrent requests
- ⬜ Reuse freed blocks when sequences complete

### Day 17 - Serving Layer (Mar 25)

- ⬜ Implement `src/llm_engine/serving/request_handler.py` (parse HTTP request -> create Request object)
- ⬜ Implement `src/llm_engine/serving/router.py` (route to scheduler, return streaming response)
- ⬜ Implement `src/llm_engine/serving/api_server.py` (FastAPI server with `/generate` endpoint)
- ⬜ Implement `src/llm_engine/serving/client.py` (Python client for testing)
- ⬜ Test end-to-end: client -> server -> generation -> streaming response

### Day 18 - Load Testing (Mar 26)

- ⬜ Implement `benchmarks/load/load_test.py` (simulate 8-32 concurrent users)
- ⬜ Implement `benchmarks/throughput/throughput_test.py`
- ⬜ Implement `benchmarks/latency/latency_test.py`
- ⬜ Test with short, medium, and long prompts
- ⬜ Measure tokens/sec throughput under load
- ⬜ Measure p95 latency under load
- ⬜ Record GPU utilization and memory consumption

### Day 19 - Full Benchmark Suite + Utils (Mar 27)

- ⬜ Implement `scripts/run_benchmark.py` (run all benchmarks end-to-end)
- ⬜ Implement `src/llm_engine/utils/logging.py` (structured logging)
- ⬜ Implement `src/llm_engine/utils/memory_utils.py` (memory tracking helpers)
- ⬜ Collect final benchmark numbers across all configurations
- ⬜ Write `tests/test_inference.py`

### Day 20 - Refinement (Mar 28)

- ⬜ Optimize batching logic
- ⬜ Improve scheduler decisions
- ⬜ Fix edge cases and bugs
- ⬜ Implement remaining `src/llm_engine/model/rotary_embedding.py` (if needed)
- ⬜ Refine feedforward implementation `src/llm_engine/model/feedforward.py` (if needed)
- ⬜ Refine transformer stack integration `src/llm_engine/model/transformer.py` (if needed)
- ⬜ Code cleanup and consistency pass

### Day 21 - Documentation + Portfolio (Mar 29)

- ⬜ Write `docs/architecture.md` (system architecture + diagrams)
- ⬜ Write `docs/design_decisions.md` (key design choices and tradeoffs)
- ⬜ Write `docs/kv_cache.md` (KV cache mechanism explanation)
- ⬜ Write `docs/scheduling.md` (continuous batching + scheduling explanation)
- ⬜ Update `README.md` (overview, architecture, features, benchmarks, how to run, future work)
- ⬜ Implement `scripts/run_server.py` (one-command server startup)
- ⬜ Include benchmark results and performance charts in docs
- ⬜ Final GitHub-ready cleanup

---

## Benchmark Checkpoints

| Checkpoint | Day | Metrics | Report |
| ---------- | --- | ------- | ------ |
| Baseline (no cache, single request) | 7 | latency, tokens/sec, GPU util | [baseline.md](baseline_benchmark.md) |
| + KV Cache | 9 | same prompts, show speedup | [kv_cache.md](kv_cache_benchmark.md) |
| + Batch Inference | 11 | 2-8 prompts, throughput gain | [batched.md](batched_kv_cache_benchmark.md) |
| + Continuous Batching | 13 | dynamic join/leave, GPU util improvement | [continuous_batching.md](continuous_batching_benchmark.md) |
| + Paged KV Cache | 16 | memory usage under concurrent requests | link |
| Full Load Test | 18 | 8-32 users, p95 latency, tokens/sec, GPU memory | link |

---

## Final Deliverables

- ⬜ Working end-to-end inference server
- ⬜ KV cache with paged memory management
- ⬜ Continuous batching scheduler
- ⬜ Benchmark suite with performance progression
- ⬜ Complete documentation with architecture diagrams
- ⬜ Clean, well-structured codebase on GitHub
