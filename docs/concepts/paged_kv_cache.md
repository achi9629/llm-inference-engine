# Day 15: Paged KV Cache — Memory Allocator

## The Problem with Contiguous Allocation

Both `KVCache` and `ContinuousKVCache` pre-allocate `max_seq_len` positions per batch slot at initialization:

```python
torch.zeros((batch_size, n_heads, max_seq_len, head_dim))  # per layer, for K and V
```

Every slot gets the full `max_seq_len` upfront, regardless of actual usage.

**Analogy: Hotel with fixed-size suites.**
Each room is a 1024-night suite — when a guest checks in, you reserve all 1024 nights even if they're staying 3 nights. The other 1021 nights sit empty but blocked.

| Sequence | Actual length | Allocated | Wasted |
|----------|--------------|-----------|--------|
| A        | 50 tokens    | 1024      | 974    |
| B        | 200 tokens   | 1024      | 824    |
| C        | 1024 tokens  | 1024      | 0      |
| D        | 10 tokens    | 1024      | 1014   |
| **Total**| **1284**     | **4096**  | **2812 (69%)** |

### Memory Cost Per Token (GPT-2 124M, fp32)

```
2 × 12 × 12 × 64 × 4 = 73,728 bytes per token

- 2  = K and V
- 12 = layers
- 12 = heads
- 64 = head_dim
- 4  = fp32 bytes
```

At `max_seq_len=1024`, `batch_size=32`: ~2.3 GB — even if most sequences are short.

---

## The Solution: Paging (Same Idea as OS Virtual Memory)

**Analogy: Hotel switches to single-night rooms.**
Instead of 1024-night suites, divide into small rooms (blocks of 16 nights). A guest staying 3 nights gets 1 room (16 nights, wastes 13). A guest staying 50 nights gets 4 rooms (64 nights, wastes 14). Rooms are returned when guests leave.

| | Contiguous Cache (current) | Paged Cache (new) |
|---|---|---|
| Unit of allocation | `max_seq_len` per sequence | Fixed-size **block** (e.g., 16 tokens) |
| When allocated | All upfront at init | On-demand as sequence grows |
| When freed | Reset entire slot or cache | Return individual blocks to free pool |
| Waste | Up to `max_seq_len - 1` per seq | Up to `block_size - 1` per seq |
| Memory layout | Contiguous per sequence | Scattered blocks, linked by table |

---

## What is the Memory Allocator?

The Memory Allocator is the **block-level free list manager**. It doesn't know about K/V tensors, layers, or attention — it only knows about numbered blocks.

**Analogy: The hotel front desk.**
The front desk maintains a list of available room numbers. When a guest needs a room, it hands out the next available number. When a guest leaves, it puts the room number back on the list. It doesn't care what happens inside the room.

```
Total blocks: [0, 1, 2, 3, 4, 5, 6, 7]   (pool of 8 blocks)

Seq A arrives, needs 3 blocks → allocate [0, 1, 2]     free: [3, 4, 5, 6, 7]
Seq B arrives, needs 1 block  → allocate [3]            free: [4, 5, 6, 7]
Seq A finishes                → free [0, 1, 2]          free: [0, 1, 2, 4, 5, 6, 7]
Seq C arrives, needs 2 blocks → allocate [0, 1]         free: [2, 4, 5, 6, 7]
```

Blocks get **reused** — Seq C gets blocks 0 and 1 that Seq A previously used.

---

## Memory Allocator API

Three operations:

| Method | What it does | Hotel analogy |
|--------|-------------|---------------|
| `allocate(num_blocks)` | Pop N block IDs from free list, return them | Guest checks in, gets room numbers |
| `free(block_ids)` | Push block IDs back onto free list | Guest checks out, rooms available again |
| `num_free_blocks` | How many blocks are still available | How many rooms are vacant |

**Data structures:**
- `num_blocks`: total number of blocks (computed from GPU memory budget)
- `free_blocks`: collection of available block IDs (starts as `[0, 1, ..., num_blocks-1]`)

The allocator is intentionally simple — just a free list. The complexity lives in the **Block Table** (Day 16).

---

## How It Connects to Existing Code

```
Day 9:  KVCache              → one seq_len (int), contiguous per-sequence
Day 13: ContinuousKVCache    → per-sequence seq_len (list), still contiguous per-sequence
Day 15: MemoryAllocator       → block-level free list (TODAY)
Day 16: BlockTable + PagedKV  → maps sequences to blocks, scattered reads/writes
```

The allocator sits **underneath** the paged cache. The paged cache calls `allocator.allocate()` when a sequence needs more space, and `allocator.free()` when a sequence finishes.

---

## Implementation Target

**File:** `src/llm_engine/cache/memory_allocator.py`

**Class:** `MemoryAllocator`

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(num_blocks: int)` | Initialize free list with all block IDs |
| `allocate` | `(num_blocks: int) -> List[int]` | Pop N blocks from free list, raise if insufficient |
| `free` | `(block_ids: List[int]) -> None` | Return blocks to free list |
| `num_free_blocks` | property | Count of available blocks |

---

## Time Complexity Analysis

### Data Structure Choice: `deque` + `set` (vLLM-style)

- `deque` for free list → O(1) `popleft()`/`append()`, preserves FIFO order for GPU memory locality
- `set` for allocated tracking → O(1) membership check, add, remove

### Per-Method Complexity

| Method | Time Complexity | Key Operations |
|--------|----------------|----------------|
| `allocate(k)` | O(k) | k × popleft O(1), k × set.add O(1) |
| `free(m)` | O(m) | m × set membership O(1), m × set.remove O(1), m × deque.append O(1) |
| `num_free_blocks` | O(1) | `len(deque)` |

### Why Not a Plain `list`?

| | `list` | `deque + set` |
|---|---|---|
| `allocate(k)` | O(k × n) — `pop(0)` shifts n elements | O(k) |
| `free(m)` | O(m × n + n log n) — `in` scan + `sort()` | O(m) |
| Double-free check | O(n) — `block_id in list` | O(1) — `block_id not in set` |

Where n = total blocks. At 1,400 blocks (70B model, A100 80GB), the list approach is ~1,400x slower per operation.

### Why `deque + set` Instead of `set` Alone?

A single `set` gives O(1) for all operations. But `set.pop()` returns an arbitrary element — block IDs come out in random order. Adding `deque` for the free list provides:

- **Determinism**: allocating 3 blocks always gives `[0, 1, 2]`, not a random set — easier to debug, test, and reproduce issues
- **Predictable reuse**: freed blocks are appended to the back and reissued in order, making allocation patterns reproducible across runs

---

# Day 16: Block Table + Paged KV Cache

## What is the Block Table?

**Analogy: Guest registry at the hotel.**
The front desk (allocator) tracks room availability. The guest registry tracks *which guest is in which rooms*. When Guest A checks in and gets rooms [0, 1, 2], the registry records:

```
Guest Registry (Block Table):
  Seq A → [0, 1, 2]     (3 blocks, ~48 tokens at block_size=16)
  Seq B → [3]            (1 block,  ~16 tokens)
  Seq C → [4, 5]         (2 blocks, ~32 tokens)
```

When Seq A finishes, the registry deletes its entry and tells the front desk to reclaim rooms [0, 1, 2].

The Block Table is a dict: `{seq_id: List[int]}` — maps sequence IDs to their ordered list of block IDs.

---

## How Tokens Map to Blocks

With `block_size=16`, token positions map to blocks like this:

```
Token positions:  [0..15]  [16..31]  [32..47]  [48..63]
                     ↓        ↓         ↓         ↓
Block index:         0        1         2         3
                     ↓        ↓         ↓         ↓
Block IDs (Seq A): [5]      [2]       [7]       [0]    ← scattered in GPU memory!
```

To find where token 35 lives:

```
block_index = 35 // 16 = 2        → 3rd block in sequence's list
offset      = 35 % 16  = 3        → 4th position within that block
block_id    = block_table[seq_id][2] = 7
GPU location = kv_cache[block_id=7][offset=3]
```

This is exactly like OS page tables: virtual address → (page number, offset) → physical address.

---

## What is the Paged KV Cache?

**Analogy: The actual hotel rooms with beds.**
The allocator tracks room numbers. The block table maps guests to rooms. The Paged KV Cache is the physical rooms — GPU tensors where K/V data is stored.

Instead of allocating per-sequence:

```python
# Old: ContinuousKVCache
torch.zeros(batch_size, n_heads, max_seq_len, head_dim)  # per layer
```

We allocate a pool of blocks:

```python
# New: PagedKVCache
torch.zeros(num_blocks, n_heads, block_size, head_dim)  # per layer, shared pool
```

The shape changes from `(batch_size, n_heads, max_seq_len, head_dim)` to `(num_blocks, n_heads, block_size, head_dim)`. The first dimension is no longer "which sequence" — it's "which block." A single sequence's data is scattered across multiple blocks.

---

## Reading and Writing with the Block Table

**Write (during decode):**
Sequence A generates a new token. Where does its K/V go?

```
seq_len = 35
block_index = 35 // 16 = 2
offset      = 35 % 16  = 3
block_id    = block_table["A"][2]  → say block 7

k_cache[layer][block_id=7, :, offset=3, :] = k_new
v_cache[layer][block_id=7, :, offset=3, :] = v_new
```

**When a block fills up (offset reaches block_size):**
Allocate a new block from the allocator and append to the sequence's block table entry.

```
seq_len = 48  → block_index = 3, but Seq A only has 3 blocks [5, 2, 7]
→ allocator.allocate(1) returns [0]
→ block_table["A"] becomes [5, 2, 7, 0]
→ write to block 0, offset 0
```

**Read (during attention):**
To compute attention, gather all K/V for Seq A by walking its block table:

```
block_table["A"] = [5, 2, 7, 0]
K_full = concat(k_cache[layer][5], k_cache[layer][2], k_cache[layer][7], k_cache[layer][0])
                  ↑ 16 tokens     ↑ 16 tokens       ↑ 16 tokens       ↑ partial
```

---

## Block Table API

| Method | What it does |
|--------|-------------|
| `add_sequence(seq_id)` | Create empty entry for a new sequence |
| `allocate_blocks(seq_id, num_blocks)` | Ask allocator for N blocks, append to sequence's entry |
| `get_block_ids(seq_id)` | Return the ordered list of block IDs for a sequence |
| `get_physical_block(seq_id, logical_index)` | Return the block ID at a given position in the sequence's list |
| `free_sequence(seq_id)` | Free all blocks for a sequence (return to allocator), delete entry |
| `num_blocks(seq_id)` | How many blocks a sequence currently holds |

## Paged KV Cache API

| Method | What it does |
|--------|-------------|
| `__init__(num_blocks, n_layers, n_heads, block_size, head_dim)` | Pre-allocate pool: `torch.zeros(num_blocks, n_heads, block_size, head_dim)` per layer |
| `write(layer_idx, block_id, offset, k, v)` | Write K/V at `cache[layer][block_id, :, offset, :]` |
| `read(layer_idx, block_ids)` | Gather K/V from scattered blocks, return concatenated tensors |
| `reset_blocks(block_ids)` | Zero out specific blocks across all layers |

---

## How They All Connect

```
Request arrives
    → ContinuousBatchingScheduler.add_request()
    → BlockTable.add_sequence(seq_id)
    → BlockTable.allocate_blocks(seq_id, num_needed)
        → MemoryAllocator.allocate(num_needed) → returns block IDs

Each decode step
    → Compute K/V for new token
    → BlockTable.get_physical_block(seq_id, token_pos // block_size)
    → PagedKVCache.write(layer, block_id, offset, k, v)
    → If block full → BlockTable.allocate_blocks(seq_id, 1)

Attention
    → BlockTable.get_block_ids(seq_id) → [5, 2, 7, 0]
    → PagedKVCache.read(layer, [5, 2, 7, 0]) → concatenated K, V

Request finishes
    → BlockTable.free_sequence(seq_id)
        → MemoryAllocator.free(block_ids)
```

---

## Day 16 Integration: Paged Cache + Continuous Batching Scheduler

### What Changed

`ContinuousBatchingScheduler` now optionally accepts `block_table` and `paged_kv_cache`. When provided, `step()` manages memory automatically:

```
step() without paging:                step() with paging:

Phase 1: Evict finished               Phase 1: Evict finished
  pop from running_requests              + block_table.free_sequence(rid)
                                         + paged_kv_cache.reset_blocks(block_ids)
                                         pop from running_requests

Phase 2: Fill empty slots              Phase 2: Fill empty slots
  pop from queue, mark RUNNING           pop from queue, mark RUNNING
                                         + block_table.add_sequence(req_id)
                                         + block_table.allocate_blocks(req_id, ceil(tokens/block_size))

Phase 3: Return active batch           Phase 3: Return active batch
```

### Responsibility Split

| Component | Responsibility | When |
|-----------|---------------|------|
| Scheduler (`step()`) | Allocate initial blocks, free blocks on evict | On enter/exit batch |
| Attention layer | `paged_kv_cache.write()` and `read()` | Each decode step |
| Scheduler does NOT | Write/read K/V data, manage offsets within blocks | Never |

The scheduler manages the **lifecycle** (birth and death of block allocations). The attention layer manages the **data flow** (writing new tokens, reading for attention).

### Block Allocation Math

```
prompt_tokens = len(req.token_ids)
num_initial_blocks = ceil(prompt_tokens / block_size)

Example: prompt = 35 tokens, block_size = 16
  → ceil(35 / 16) = 3 blocks allocated on entry
  → block 0: tokens 0-15
  → block 1: tokens 16-31
  → block 2: tokens 32-34 (+ 13 empty slots for future decode tokens)
```

### Test Verification

`test_continuous_batching_with_paged_cache` in test_scheduler.py:

- 3 requests, 8 tokens each, block_size=4 → 2 blocks per request
- Step 1: 2 requests enter → 4 blocks allocated, 12 free
- Step 2: complete one, step → freed 2 blocks, new request gets 2 → still 12 free
- Step 3: complete remaining, step → all freed, 16 free, has_work=False

---

## Day 16: Forward Pass Integration — Paged Cache + Attention Layer

### The Problem: Interface Mismatch

The current attention layer (`attention.py`, line 78) has a simple contract with the KV cache:

```
k, v = kv_cache.update_cache(layer_idx, k, v)
```

`update_cache()` does two jobs in one call:
1. **Stores** the new K/V tokens contiguously in a pre-allocated slab
2. **Returns** the full K/V history (old + new) as a batch tensor of shape `(B, n_heads, T_total, head_dim)`

The attention layer then uses `k` and `v` directly for `Q @ K^T` — no further thought needed.

`PagedKVCache` has a fundamentally different interface:
- `write()`: stores one token at a time to a specific block + offset
- `read()`: gathers scattered blocks for one sequence, returns `(1, n_heads, num_blocks * block_size, head_dim)`

Three mismatches:

| | `update_cache()` (contiguous) | `write()`/`read()` (paged) |
|---|---|---|
| Granularity | Whole batch at once | Per-sequence |
| Metadata needed | Just `layer_idx` | `block_id`, `offset`, `seq_id` |
| Output | Stacked batch `(B, ...)` | Single sequence `(1, ...)` |

---

### The Adapter Pattern (Hotel Concierge Analogy)

**Analogy: Hiring a concierge between the guest's assistant and the hotel.**

The guest's personal assistant (attention layer) has been trained to work with a single-building apartment (contiguous cache) — they just say "store my stuff on the next shelf" and "give me all my stuff."

Rather than retraining the assistant, you hire a **concierge** — someone who sits between the assistant and the hotel, translating requests:

- **Assistant says "store this"** → Concierge looks up the guest registry, finds the right room number and shelf position, stores it in the hotel
- **Assistant says "give me everything"** → Concierge looks up all rooms for this guest, retrieves items from each room, arranges them in order, hands back a neat stack

This concierge is a **wrapper object** (`PagedCacheContext`) that bundles:
- The `paged_kv_cache` (the hotel)
- The `block_table` (the guest registry)
- The current batch's sequence IDs and token positions (who's checking in right now)

It exposes the same `update_cache(layer_idx, k, v)` interface, so the attention layer doesn't change at all — **duck typing** handles the rest.

---

### Data Flow Inside `update_cache()` (Adapter)

During one decode step with a batch of 3 sequences:

**Write phase:**
For each sequence `i` in the batch:
```
1. seq_id → block_table.get_block_ids(seq_id) → e.g. [5, 12, 3]
2. token_position = number of tokens generated so far
3. block_index = token_position // block_size    → which block in the list
4. offset      = token_position % block_size     → which slot within that block
5. block_id    = block_ids[block_index]          → physical block ID
6. paged_kv_cache.write(layer_idx, block_id, offset, k[i], v[i])
```

**Read phase:**
For each sequence `i` in the batch:
```
1. block_ids = block_table.get_block_ids(seq_id) → [5, 12, 3]
2. k_seq, v_seq = paged_kv_cache.read(layer_idx, block_ids)
   → shape: (1, n_heads, num_blocks * block_size, head_dim)
3. Slice to valid tokens: k_seq[:, :, :actual_seq_len, :]
```

Stack all sequences into `(B, n_heads, max_T_total, head_dim)`, padding shorter ones.

---

### The Trailing-Zeros Problem

`read()` returns ALL slots in ALL blocks — including empty/unused slots in the last block.

```
Sequence with 5 tokens, block_size=4:
  Block 0: [t0, t1, t2, t3]    ← fully used
  Block 1: [t4,  0,  0,  0]    ← 3 empty slots

read() returns 8 positions: [t0, t1, t2, t3, t4, 0, 0, 0]
```

Those trailing zeros would **corrupt attention scores** — they appear as valid "past" positions (not future, so causal mask won't help). The zeros get multiplied with Q, producing small but non-zero attention values directed at empty positions.

**Solution: Slice after read.**

```
k_seq = k_seq[:, :, :actual_seq_len, :]   # (1, n_heads, 5, head_dim)
```

This requires tracking per-sequence token count, which the adapter already has via the batch metadata.

---

### Files That Need to Change

| File | Change | Reason |
|---|---|---|
| **New: adapter/wrapper class** | Bundles paged_kv_cache + block_table + batch metadata. Exposes `update_cache()` that internally calls write + read + stack. Also needs `seq_len` property for position embedding offset in transformer.py. | Isolates all paged logic behind the same interface |
| **generator.py** | Create adapter wrapper each step with current batch's sequence metadata. Pass it as the `kv_cache` argument. | Generator knows which sequences are in the batch |
| **attention.py** | **No changes** | Duck typing — calls `kv_cache.update_cache(layer_idx, k, v)` as before |
| **block.py** | **No changes** | Just pipes `kv_cache` through |
| **transformer.py** | **No changes** | Just pipes `kv_cache` through, reads `kv_cache.seq_len` for position offset |

### Why This Design?

The adapter pattern completely isolates the paged complexity:

- The model code (`attention.py`, `block.py`, `transformer.py`) stays **unchanged**
- All paged logic lives in the wrapper + the cache/block_table modules already built
- Existing tests for attention, KVCache, ContinuousKVCache continue to pass — no backward compatibility break
- Switching between contiguous and paged cache is just a matter of which object you pass as `kv_cache`
