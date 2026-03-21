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
