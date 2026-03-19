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

- 7 prompts created as Request objects
- 3 added to scheduler, 2 scheduled (max_batch=2)
- 2 more added, 1 completed
- Asserts: request 0 = finished, request 1 = running, request 2 = pending
