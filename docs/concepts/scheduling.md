# Day 12: Request Queue + Batch Scheduler

## Why Is This Needed?

Up to Day 11, our inference engine works like this:

```bash
User --> InferenceEngine.generate("prompt") --> tokens --> done
```

This is **monolithic** -- one prompt in, one result out. It can't handle:

- Multiple users sending prompts at different times
- Deciding which prompts to batch together
- Tracking which requests are waiting, running, or finished

To build a **server** that handles concurrent users, we need to decompose generation into **schedulable units**.

## The Three Components

### 1. Request (request.py) -- The Data Container

A Request tracks one user's prompt through its entire lifecycle.

| Field            | Purpose                               |
|------------------|---------------------------------------|
| request_id       | Unique identifier                     |
| prompt           | Original text (for returning to user) |
| token_ids        | Tokenized prompt (for the model)      |
| max_tokens       | Generation limit                      |
| generated_tokens | Tokens produced so far (starts empty) |
| state            | PENDING -> RUNNING -> FINISHED/FAILED |
| created_at       | Timestamp for ordering/metrics        |

### 2. RequestQueue (request_queue.py) -- The Waiting Line

A FIFO queue holding Request objects waiting to be scheduled.

| Method                    | What It Does                              |
|---------------------------|-------------------------------------------|
| add(request)              | Append to back of queue                   |
| pop()                     | Remove from front (returns None if empty) |
| get_batch(max_batch_size) | Pop up to N requests at once              |
| size()                    | Number of waiting requests                |
| is_empty()                | Check if queue is empty                   |

### 3. BatchScheduler (batch_scheduler.py) -- The Decision Maker

Pulls requests from the queue, groups them into batches, and tracks state transitions.

| Method                 | What It Does                       |
|------------------------|------------------------------------|
| add_request(request)   | Add to internal queue              |
| schedule()             | Pop batch from queue, mark RUNNING |
| complete_request(id)   | Mark FINISHED, move to completed   |
| get_running_requests() | List currently generating requests |

## State Machine

```bash
PENDING --> RUNNING --> FINISHED
                   \-> FAILED
```

- **PENDING**: Request is in the queue, waiting to be picked
- **RUNNING**: Request is part of an active generation batch
- **FINISHED**: Generation completed successfully
- **FAILED**: Something went wrong (timeout, error, etc.)

## Data Flow

```bash
User sends prompt
       |
       v
+-------------+
|  Tokenizer  |  encode("Tell me a joke") -> [24446, 502, 257, 9707]
+------+------+
       |
       v
+-------------+
|   Request    |  Request(id, prompt, token_ids, max_tokens)
|   Created    |  state = PENDING
+------+------+
       |
       v
+-------------+
| RequestQueue |  queue: [req_A, req_B, req_C, ...]
|  (FIFO)     |
+------+------+
       |  scheduler.schedule()
       v
+--------------+
|BatchScheduler|  Pops up to max_batch_size requests
|              |  Marks them RUNNING
+------+-------+
       |
       v
+-------------+
|  Generator   |  Runs batched inference on token_ids
|  (existing)  |  Produces generated_tokens
+------+------+
       |
       v
+-------------+
|  Complete    |  scheduler.complete_request(id)
|  Request     |  state = FINISHED
+-------------+
```

## Example Walkthrough

```bash
t=0  Add requests A, B, C to scheduler     queue: [A, B, C]      all PENDING
t=1  scheduler.schedule(max_batch=2)        queue: [C]            A,B = RUNNING
t=2  Add requests D, E                      queue: [C, D, E]      D,E = PENDING
t=3  complete_request(A)                    A = FINISHED          B still RUNNING
t=4  scheduler.schedule(max_batch=2)        queue: [E]            C,D = RUNNING
```

## Why Pre-Tokenize in Request?

The scheduler needs token_ids (not raw strings) to:

1. **Calculate padding** -- batch requests with similar lengths to minimize waste
2. **Estimate memory** -- longer sequences use more KV cache memory
3. **Make scheduling decisions** -- fit as many requests as GPU memory allows

Tokenization moves from inside generate() to when the request arrives.

## How This Connects to Future Days

| Day       | Component                   | Uses                                              |
|-----------|-----------------------------|---------------------------------------------------|
| Day 12    | Request + Queue + Scheduler | today                                             |
| Day 13    | Continuous Batching         | Adds/removes requests mid-generation              |
| Day 15-16 | Paged KV Cache              | Memory-aware scheduling                           |
| Day 17    | FastAPI Server              | HTTP -> Request -> Queue -> Scheduler -> Response |

## Test Verification

test_scheduler.py simulates the full flow:

- 6 prompts created as Request objects
- 3 added to scheduler, 2 scheduled (max_batch=2)
- 2 more added, 1 completed
- Asserts: request 0 = finished, request 1 = running, request 2 = pending

---

## Day 13: Continuous Batching + Per-Sequence KV Cache

## Why Continuous Batching?

Static batching (BatchScheduler) has a fundamental waste problem:

```bash
Static Batching (batch_size=4):

Step 1:  [A, B, C, D]   all running
Step 10: [A, B, C, D]   A finishes early (hit EOS at step 10)
Step 11: [_, B, C, D]   slot 0 is WASTED -- GPU computes padding for empty slot
Step 20: [_, _, C, D]   B finishes -- now 2 slots wasted
Step 30: [_, _, _, D]   only D running -- 75% GPU waste
Step 40: [_, _, _, _]   D finishes -- NOW we can start next batch
```

**ML analogy:** Like training with a DataLoader that waits for the slowest sample in the batch before loading the next batch. Continuous batching is like replacing finished samples immediately.

```bash
Continuous Batching (batch_size=4):

Step 1:  [A, B, C, D]   all running
Step 10: [E, B, C, D]   A finishes -> E fills slot 0 immediately
Step 20: [E, F, C, D]   B finishes -> F fills slot 1 immediately
Step 30: [E, F, G, D]   C finishes -> G fills slot 2 immediately
```

GPU stays full at all times.

## ContinuousBatchingScheduler API

| Method               | What It Does                                          |
|----------------------|-------------------------------------------------------|
| add_request(request) | Add to internal queue                                 |
| step()               | Evict finished, fill empty slots, return active batch |
| complete_request(id) | Mark request as FINISHED                              |
| get_active_batch()   | List currently active requests (read-only)            |
| has_work()           | True if queue has pending or running requests         |

## How step() Works -- The 3-Phase Loop

```bash
step() called each generation iteration:

Phase 1: EVICT
  for each request in running_requests:
    if request.is_finished() -> remove from running_requests

Phase 2: FILL
  num_empty = max_batch_size - len(running_requests)
  pop up to num_empty requests from queue
  mark them RUNNING, add to running_requests

Phase 3: RETURN
  return list(running_requests.values())
```

## Static vs Continuous -- Side by Side

| Aspect              | BatchScheduler (Static)          | ContinuousBatchingScheduler      |
|---------------------|----------------------------------|----------------------------------|
| When to refill      | After entire batch finishes      | Every step                       |
| GPU utilization     | Degrades as requests finish      | Stays near 100%                  |
| Key method          | schedule() (one-time)            | step() (called every iteration)  |
| Request replacement | All-or-nothing                   | Individual slot replacement      |
| Queue drain rate    | Bursty (batch at a time)         | Smooth (one slot at a time)      |

## ContinuousKVCache -- Per-Sequence Tracking

The original `KVCache` tracks a single `seq_len` (int) for the whole batch -- all sequences share the same position pointer. This breaks with continuous batching because:

- Sequence A might be at position 50 (nearly done)
- Sequence E just joined at position 0 (just started)

`ContinuousKVCache` fixes this:

| Feature          | KVCache (static)            | ContinuousKVCache (continuous)      |
|------------------|-----------------------------|-------------------------------------|
| seq_len          | Single int                  | List[int], one per batch slot       |
| update_cache()   | Same start pos for all      | Per-batch-idx start positions       |
| reset            | reset_cache() (zeros all)   | reset_slot(batch_idx) (zeros one)   |
| increment_seq_len| (layer_idx, T_new)          | (batch_idx, layer_idx, T_new)       |

### reset_slot() -- The Key Operation

When a sequence finishes and a new one takes its slot:

```bash
Before reset_slot(0):
  slot 0: seq_len=50, cache filled with 50 tokens of old sequence
  slot 1: seq_len=30, cache filled with 30 tokens (still running)

After reset_slot(0):
  slot 0: seq_len=0,  cache ZEROED (ready for new sequence)
  slot 1: seq_len=30, cache UNTOUCHED (still running)
```

## Test Verifications

### test_scheduler.py (2 tests)

- **test_scheduler**: BatchScheduler flow -- add, schedule, complete, assert states
- **test_continuous_batching_step**: 4-phase test -- fill slots, evict+refill, drain, empty

### test_continuous_kv_cache.py (5 tests)

- **test_initialization**: seq_len is [0, 0], shapes correct, all zeros
- **test_per_sequence_seq_len**: seq_len only updates on last layer
- **test_reset_slot**: zeros one slot, other slot untouched
- **test_reset_cache**: full reset, everything zeroed
- **test_update_cache_per_batch_idx**: prefill 3 + decode 1, verify positions match
