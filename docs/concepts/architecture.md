# System Architecture

End-to-end LLM inference engine built from scratch in PyTorch.
The engine is **model-agnostic** — the model layer is a pluggable slot behind
a standard `nn.Module` interface. GPT-2 124M is the current implementation;
new families (Mistral, Falcon, …) plug in without touching the layers above it.

---

## Component Diagram

```bash
┌─────────────────────────────────────────────────────────┐
│                     HTTP Layer                          │
│  FastAPI  ·  POST /generate  ·  GET /health             │
│  Semaphore (max_concurrent_requests)  ·  Timeout guard  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   Serving Layer                         │
│  Router (sync / async)  ·  RequestHandler               │
│  Validates prompt → tokenizes → creates Request object  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Scheduling Layer                       │
│  ContinuousBatchingScheduler                            │
│  RequestQueue (FIFO deque)  ·  Request state machine    │
│  PENDING → RUNNING → FINISHED                           │
│  Fills empty batch slots each step; frees on completion │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Inference Layer                        │
│  InferenceEngine  ·  generator()  ·  greedy_sampler()   │
│  Autoregressive loop: forward → logits → sample → append│
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    Model Layer                          │
│  nn.Module with forward(input_ids, position_ids,        │
│                         kv_cache, padding_mask)         │
│  ┌─────────────────────────────────────────────┐        │
│  │  GPT-2 124M  (current)                      │        │
│  │  Embedding → 12× TransformerBlock → LM Head │        │
│  └─────────────────────────────────────────────┘        │
│  Future: Mistral, Falcon — same interface                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    Cache Layer                          │
│  KVCache (simple)  ·  ContinuousKVCache (per-slot)      │
│  PagedKVCache + BlockTable + MemoryAllocator            │
│  All caches expose: update_cache(layer_idx, k, v)       │
└─────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Sync Mode (`async_mode=False`)

```bash
Client ──POST /generate──▶ api_server
                              │
                    acquire semaphore (503 if full)
                              │
                        Router.generate()
                              │
                   RequestHandler.handle()
                   ├─ tokenizer.encode(prompt)
                   └─ Request(id, prompt, token_ids, max_tokens)
                              │
                   scheduler.add_request(request)
                   scheduler.step()  ← fills batch slot
                              │
                   InferenceEngine.generate()
                   ├─ model.forward(token_ids, kv_cache)
                   ├─ greedy_sampler(logits)
                   └─ loop until EOS or max_tokens
                              │
                   scheduler.complete_request()
                   ├─ request.state = FINISHED
                   └─ block_table.free(seq_id)  [paged mode]
                              │
                    release semaphore
                              │
Client ◀──JSON response───────┘
```

### Async Mode (`async_mode=True`)

```bash
Client ──POST /generate──▶ api_server
                              │
                        Router.generate()
                   ├─ enqueue Request
                   └─ return Future
                              │
              ┌───────────────┘
              ▼
     Background generation loop
     ├─ scheduler.step()       ← continuous batching
     ├─ engine.generate(batch) ← batch of active requests
     └─ scheduler.complete_request() per finished request
              │
              ▼
     Future resolved → response returned to client
```

---

## Source Map

29 source files across 8 packages.

| Package      | Files                                                                                                                           | Responsibility                                                    |
|--------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `serving/`   | `api_server.py`, `router.py`, `request_handler.py`, `client.py`                                                                 | HTTP endpoints, request routing, sync/async orchestration         |
| `scheduler/` | `continuous_batching.py`, `batch_scheduler.py`, `request.py`, `request_queue.py`                                                | Batch formation, request lifecycle (PENDING → RUNNING → FINISHED) |
| `inference/` | `inference_engine.py`, `generator.py`, `sampler.py`                                                                             | Autoregressive token generation, sampling strategies              |
| `model/`     | `GPT2/{transformer, block, attention, feedforward, conv}.py`, `__init__.py`                                                     | Model implementations; `load_model()` dispatches by family        |
| `cache/`     | `kv_cache.py`, `continuous_kv_cache.py`, `paged_kv_cache.py`, `memory_allocator.py`, `block_table.py`, `paged_cache_context.py` | KV cache management (simple, continuous, paged)                   |
| `tokenizer/` | `tokenizer.py`                                                                                                                  | Encode/decode via HuggingFace tokenizer                           |
| `config/`    | `config_loader.py`                                                                                                              | YAML config loading (model, scheduler, server)                    |
| `utils/`     | `weight_loader.py`, `profiler.py`, `gpu_monitor.py`                                                                             | Weight loading, torch profiler, GPU memory tracking               |

---

## Configuration

Three YAML files under `configs/`:

| File                    | Key Fields                                                                     | Purpose                                 |
|-------------------------|--------------------------------------------------------------------------------|-----------------------------------------|
| `model_config.yaml`     | `model_family`, `model_variant`, `model_dir`, `max_seq_len`, `device`, `dtype` | Which model to load and where           |
| `scheduler_config.yaml` | `max_batch_size`, `max_tokens_per_batch`, `scheduling_policy`                  | Batch size limits, scheduling strategy  |
| `server_config.yaml`    | `host`, `port`, `timeout`, `max_concurrent_requests`                           | Uvicorn bind address, concurrency guard |

---

## Model Interface Contract

Any model plugged into the engine must satisfy:

```python
class LLM(nn.Module):
    def forward(self,
                input_ids: torch.Tensor,           # [B, T]
                position_ids: torch.Tensor,        # [B, T]
                kv_cache: KVCache | None,          # duck-typed cache
                padding_mask: torch.Tensor | None  # [B, T]
    ) -> torch.Tensor:                             # logits [B, T, vocab_size]
```

The KV cache is duck-typed — any object with
`update_cache(layer_idx, k, v) → (k_full, v_full)` works.
This means `KVCache`, `ContinuousKVCache`, and `PagedCacheContext` are all
interchangeable from the model's perspective.

---

## Adding a New Model

1. Create `src/llm_engine/model/<Family>/transformer.py` implementing
   the interface above.
2. Update `load_model()` in `src/llm_engine/model/__init__.py` to
   dispatch on `model_family`.
3. Add a `configs/model_config.yaml` variant
   (e.g. `model_family: "mistral"`, `model_variant: "7B"`).
4. Place weights under `assets/models/<family>/<variant>/`.

No changes needed to serving, scheduling, inference, or cache layers.

---

## Test Coverage

122 tests across 15 test files covering every layer from tokenizer
through API server. Tests run on both CPU and CUDA.
