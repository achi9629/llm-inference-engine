# Day 18: Sync vs Async Router

## Why the Sync Router Can't Batch

The Day 17 Router processes one request at a time:

```bash
Request A arrives  →  handle → add → step → generate → complete → return
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      Request B BLOCKED until A finishes
```

When 4 clients POST simultaneously, they execute **sequentially**. The `ContinuousBatchingScheduler` (max_batch_size=4) never sees more than 1 request because each is fully processed before the next enters.

## The Async Solution

Split the Router into two parts:

1. **`generate()` becomes async** — adds request to scheduler, then `await`s a Future
2. **`_generation_loop()`** — background task that drains the scheduler continuously

### Before (Sync)

```bash
generate() {
    handle → add_request → step → engine.generate(single) → complete → return
}
```

### After (Async)

```bash
generate():                           _generation_loop() (background):
    handle → add_request              while True:
    future = create Future                if scheduler.has_work():
    await future  ←─── notified ──────      batch = scheduler.step()
    return result                            for req in batch:
                                                result = engine.generate(req)
                                                complete_request(req)
                                                future.set_result(result) ──→ notify
                                         else:
                                             await asyncio.sleep(0.01)
```

## How Concurrent Batching Works

When 4 clients POST at the same time:

1. FastAPI accepts all 4 as concurrent coroutines (async)
2. Each `generate()` adds its request to the scheduler and `await`s a Future
3. The background loop wakes up, sees 4 pending requests
4. `scheduler.step()` returns a batch of up to `max_batch_size` requests
5. `engine.generate()` runs for each request in the batch
6. All 4 Futures resolve → all 4 clients get responses

The scheduler finally does what it was designed for — **batch multiple requests together**.

## New Data Structures

| Field              | Type                        | Purpose                                              |
|--------------------|-----------------------------|------------------------------------------------------|
| `_pending_futures` | `dict[str, asyncio.Future]` | Maps request_id → Future (the waiter)                |
| `_loop_task`       | `asyncio.Task`              | Reference to the background generation loop          |
| `_lock`            | `asyncio.Lock`              | Protects scheduler access from concurrent coroutines |

## Key Design Decisions

### Why `asyncio.Lock`?

Multiple `generate()` coroutines can call `scheduler.add_request()` concurrently. The `_generation_loop()` reads from the same scheduler. The lock prevents race conditions between adding and reading requests.

### Why `asyncio.sleep(0.01)` in the idle loop?

Without it, the loop busy-spins and starves other coroutines. The 10ms sleep yields control to the event loop, letting new `generate()` calls proceed.

### Why is `engine.generate()` blocking acceptable?

GPU inference is inherently blocking — the CPU waits for CUDA kernels. The background loop holds the lock during inference, which is correct: only one generation pass should use the GPU at a time. Other coroutines can still add requests to the scheduler while waiting for the lock.

## Files Changed

| File            | Change                                                                                              |
|-----------------|-----------------------------------------------------------------------------------------------------|
| `router.py`     | `asyncio` imports, `_pending_futures`, `_lock`, `start()`, async `generate()`, `_generation_loop()` |
| `api_server.py` | Endpoint becomes `async def`, `await router.generate()`, startup hook for `router.start()`          |
| `run_server.py` | No change (uvicorn already runs an event loop)                                                      |

## Comparison

| Aspect                | Sync (Day 17)         | Async (Day 18)                             |
|-----------------------|-----------------------|--------------------------------------------|
| Requests processed    | One at a time         | Multiple concurrently                      |
| Scheduler utilization | Always batch_size=1   | Up to max_batch_size                       |
| GPU utilization       | Idle between requests | Better — batched inference                 |
| Complexity            | Simple                | Moderate (futures, locks, background task) |
| Use case              | Testing, single user  | Production, multi-user                     |
