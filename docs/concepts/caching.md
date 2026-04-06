# KV Cache

## 1. The Problem: Redundant Computation in Autoregressive Decoding

Transformer attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

During autoregressive generation, the model produces one token at a time. Without caching, **every decode step re-encodes the entire sequence** — recomputing K and V projections for all previous tokens. For a sequence of length $n$ generating $T$ new tokens, total attention cost is:

$$\sum_{t=1}^{T} O\!\left((n + t)^2 \cdot d\right) \approx O(T \cdot n^2 \cdot d)$$

With KV cache, past K/V projections are stored and reused. Each decode step only computes K/V for the **single new token** and reads the cached history:

$$\sum_{t=1}^{T} O\!\left((n + t) \cdot d\right) \approx O(T \cdot n \cdot d)$$

This reduces decode-phase attention from $O(n^2)$ per step to $O(n)$ per step — a fundamental optimization used by every production inference system.

**Benchmark evidence (GPT-2 124M, A100, single request, 50 generated tokens):**

| Prompt Len | No Cache (tok/s) | KV Cache (tok/s)  | Speedup |
|------------|------------------|-------------------|---------|
| 64         | 180.3            | 180.7             | 1.00x   |
| 256        | 126.2            | 179.4             | 1.42x   |
| 512        | 80.0             | 174.9             | 2.19x   |

KV cache latency stays flat (~0.28s) regardless of prompt length. Without cache, latency grows quadratically.

---

## 2. Cache Implementations

This engine has three KV cache variants, each solving a different problem. All expose the same duck-typed interface so the attention layer needs no code changes:

```python
kv_cache.update_cache(layer_idx, k, v) → (k_full, v_full)
kv_cache.seq_len → int
```

### 2.1 Standard KV Cache (`KVCache`)

**File:** `src/llm_engine/cache/kv_cache.py`

Pre-allocates fixed tensors per layer at init:

```python
k_cache[layer] = zeros(batch_size, n_heads, max_seq_len, head_dim)
v_cache[layer] = zeros(batch_size, n_heads, max_seq_len, head_dim)
```

`update_cache` writes new K/V into the next available positions using a global `seq_len` counter:

```python
start = self.seq_len
end   = start + k.shape[2]  # T_new (prompt_len on prefill, 1 on decode)

self.k_cache[layer_idx][:, :, start:end, :] = k
self.v_cache[layer_idx][:, :, start:end, :] = v

return self.k_cache[layer_idx][:, :, :end, :],
       self.v_cache[layer_idx][:, :, :end, :]
```

**Properties:**

- Simple contiguous memory layout — efficient tensor slicing
- Fixed `batch_size` at init: all requests in a batch must be present from the start
- Cannot handle dynamic batching (tensor shape mismatch if batch size changes)
- Used for: standalone `engine.generate()`, single-request serving, benchmarks

**Limitation:** `seq_len` is a single scalar shared across the batch. This works when all sequences have the same length (left-padded prompts start generating simultaneously). It cannot track per-sequence positions independently.

### 2.2 Continuous KV Cache (`ContinuousKVCache`)

**File:** `src/llm_engine/cache/continuous_kv_cache.py`

Same tensor layout as `KVCache`, but tracks **per-sequence** positions:

```python
self.seq_len = [0] * batch_size  # per-sequence counter (not a single scalar)
```

`update_cache` writes each batch element at its own position:

```python
for batch_idx in range(self.batch_size):
    seq_len = self.seq_len[batch_idx]
    start = seq_len
    end = seq_len + k.size(2)
    self.k_cache[layer_idx][batch_idx, :, start:end, :] = k[batch_idx]
    self.v_cache[layer_idx][batch_idx, :, start:end, :] = v[batch_idx]
```

**Key addition — `reset_slot(batch_idx)`:**

```python
def reset_slot(self, batch_idx):
    self.seq_len[batch_idx] = 0
    for layer_idx in range(self.n_layers):
        self.k_cache[layer_idx][batch_idx, :, :, :] = 0
        self.v_cache[layer_idx][batch_idx, :, :, :] = 0
```

This enables **continuous batching**: when sequence `i` finishes, its slot is zeroed and reused by a new request — without recreating the entire cache.

**Properties:**

- Per-sequence tracking enables slot reuse for continuous batching
- Still pre-allocates full `(batch_size, n_heads, max_seq_len, head_dim)` per layer
- Batch size is still fixed at init — the maximum concurrent sequences is bounded by `batch_size`
- Memory is allocated for `max_seq_len` even if most sequences are short

### 2.3 Paged KV Cache (Block-Level Memory)

**Files:** `src/llm_engine/cache/paged_kv_cache.py`, `memory_allocator.py`, `block_table.py`, `paged_cache_context.py`

Inspired by OS virtual memory. Instead of pre-allocating contiguous per-sequence buffers, the paged cache maintains a **shared pool of fixed-size blocks**:

```python
k_cache[layer] = zeros(num_blocks, n_heads, block_size, head_dim)
v_cache[layer] = zeros(num_blocks, n_heads, block_size, head_dim)
```

Sequences don't own contiguous memory — their K/V data is scattered across blocks, mapped through a **block table** (analogous to a page table).

#### Components

```bash
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────────────┐
│  MemoryAllocator │      │    BlockTable     │      │     PagedKVCache         │
│                  │      │                   │      │                          │
│  free_blocks:    │◄────►│  seq_id → [b0,    │─────►│  k_cache[layer][block]   │
│    deque(0..N)   │      │           b1, b2] │      │  v_cache[layer][block]   │
│  allocated: set  │      │                   │      │                          │
└──────────────────┘      └──────────────────┘      └──────────────────────────┘
         │                         │                            │
         └──── allocate/free ──────┘                            │
                                                                │
                              ┌──────────────────────────┐      │
                              │   PagedCacheContext       │──────┘
                              │   (Adapter)               │
                              │                           │
                              │   update_cache(l, k, v)   │
                              │   → write + read + pad    │
                              └──────────────────────────┘
```

**MemoryAllocator** — Manages block IDs as a FIFO free list (`collections.deque`) + allocated set. Agnostic to KV tensors — only tracks integer block IDs. O(k) allocate, O(m) free.

**BlockTable** — Maps `seq_id → [block_id_0, block_id_1, ...]`. Acts as the page table:

```python
block_index = token_position // block_size
offset      = token_position % block_size
block_id    = block_table[seq_id][block_index]
```

Delegates allocation/deallocation to `MemoryAllocator`.

**PagedKVCache** — The physical block pool. Supports:

- `write(layer_idx, block_id, offset, k, v)` — single-token or batched writes
- `read(layer_idx, block_ids, max_blocks)` — gather + concat blocks for attention
- `reset_blocks(block_ids)` — zero out blocks when sequences finish

**PagedCacheContext** — The adapter that wraps the above components behind the standard `update_cache(layer_idx, k, v)` interface. Called by the attention layer identically to `KVCache`:

1. **Write phase**: Computes `(block_id, offset)` for each new token, issues a single batched write
2. **Read phase**: Gathers all blocks per sequence via `get_block_ids_for_batch`, calls batched `read`, slices to valid token count
3. **Increment**: Updates per-sequence `seq_lens` after the last layer

**Properties:**

- Memory scales with actual usage, not worst-case `max_seq_len`
- No batch dimension in storage — any number of sequences can coexist
- Enables dynamic batching (batch size can vary every scheduling step)
- Near-zero internal fragmentation (waste ≤ `block_size - 1` tokens per sequence)
- Block reuse: finished sequences return blocks to the pool immediately

---

## 3. Comparison

| Property              | KVCache                     | ContinuousKVCache             | PagedKVCache               |
|-----------------------|-----------------------------|-------------------------------|----------------------------|
| Tensor layout         | `(B, H, T, D)`              | `(B, H, T, D)`                | `(num_blocks, H, S, D)`    |
| Batch size            | Fixed at init               | Fixed at init                 | Dynamic (any B)            |
| Per-sequence tracking | No (global)                 | Yes (`seq_len[]`)             | Yes (`seq_lens[]`)         |
| Slot reuse            | No — full reset             | Yes — `reset_slot`            | Yes — free blocks          |
| Memory allocation     | Upfront worst-case          | Upfront worst-case            | On-demand per block        |
| Internal fragmentation| Up to `max_seq_len` per seq | Up to `max_seq_len` per seq   | ≤ `block_size - 1` per seq |
| Dynamic batching      | No                          | No                            | Yes                        |
| Used in               | Standalone/benchmarks       | Continuous batching scheduler | Production serving         |

Where `B` = batch_size, `H` = n_heads, `T` = max_seq_len, `D` = head_dim, `S` = block_size.

---

## 4. Memory Benchmark (GPT-2 124M, A100, batch_size=4, block_size=16, 50 tokens)

| num_blocks | Standard Peak (MB) | Paged Peak (MB) | Memory Savings |
|------------|--------------------|-----------------|----------------|
| 64         | 966.8              | 620.2           | **35.9%**      |
| 128        | 972.7              | 692.2           | **28.8%**      |
| 256        | 972.7              | 836.2           | **14.0%**      |

Standard memory is constant (~970 MB) — pre-allocates `(batch, heads, n_ctx=1024, d_head)` regardless of actual sequence lengths. Paged memory scales linearly with block count, allocating only what's needed.

**Throughput tradeoff:**

| num_blocks | Standard (tok/s) | Paged (tok/s) | Overhead |
|------------|------------------|---------------|----------|
| 64         | ~565             | ~435          | -23%     |
| 128        | ~586             | ~410          | -30%     |
| 256        | ~583             | ~436          | -25%     |

Paged cache has **~25% throughput overhead** per-batch due to gather/scatter block indexing vs. contiguous slicing. This is the standard tradeoff — paged cache wins on **system throughput** (more concurrent sequences) while losing on **per-batch throughput** (index_select vs. slice).

**Serving throughput (concurrent requests, GPT-2 124M, A100, max_tokens=50):**

| Metric (c=64)     | Standard (sequential) | Paged (batched) | Speedup |
|-------------------|-----------------------|-----------------|---------|
| Short throughput  | 165 tok/s             | 532 tok/s       | 3.2x    |
| Medium throughput | 166 tok/s             | 455 tok/s       | 2.7x    |
| Long throughput   | 167 tok/s             | 234 tok/s       | 1.4x    |

Under concurrent load, paged cache + continuous batching delivers **1.4–3.2x higher system throughput** because multiple sequences share the GPU simultaneously.

---

## 5. How It Connects to Attention

The attention layer (`MultiHeadAttention.forward`) is cache-agnostic:

```python
# In attention.py forward():
q, k, v = ...  # project from input: each (B, n_heads, T_new, head_dim)

if kv_cache is not None and layer_idx is not None:
    k, v = kv_cache.update_cache(layer_idx, k, v)  # duck-typed call
    T_total = k.size(2)                              # includes cached history
else:
    T_total = T_new

score = (q @ k.transpose(-2, -1)) * self.scale      # (B, H, T_new, T_total)
```

During **prefill** (first forward pass): `T_new = prompt_length`. The full prompt's K/V is written to the cache.

During **decode** (subsequent steps): `T_new = 1`. Only the new token's K/V is written. The cache returns the full history so attention sees all previous tokens.

This is why `seq_len` tracking matters — it tells `update_cache` where to write the new token. And why `increment_seq_len` only fires on the last layer: all layers process the same input in a single forward pass, so `seq_len` should increment once per step, not once per layer.

---

## 6. Source Map

| File                           | Class               | Purpose                                                    |
|--------------------------------|---------------------|------------------------------------------------------------|
| `cache/kv_cache.py`            | `KVCache`           | Standard pre-allocated cache, fixed batch size             |
| `cache/continuous_kv_cache.py` | `ContinuousKVCache` | Per-sequence tracking + slot reuse                         |
| `cache/paged_kv_cache.py`      | `PagedKVCache`      | Block pool: write, read, reset                             |
| `cache/memory_allocator.py`    | `MemoryAllocator`   | Free list + allocated set for block IDs                    |
| `cache/block_table.py`         | `BlockTable`        | Seq → block ID mapping (page table)                        |
| `cache/paged_cache_context.py` | `PagedCacheContext` | Adapter: wraps paged cache behind `update_cache` interface |
