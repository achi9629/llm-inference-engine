# Day 17: Serving Layer

## What It Is

The serving layer is the HTTP boundary between external users and the inference engine. It turns a `curl` request into generated text.

```bash
HTTP POST /generate  →  RequestHandler  →  Router  →  Scheduler  →  InferenceEngine  →  Response
```

## Why It Matters

The project currently has two independent paths:

| Path        | Entry Point                          | How It Works                                                                |
|-------------|--------------------------------------|-----------------------------------------------------------------------------|
| **Offline** | `InferenceEngine.generate()`         | Direct Python call, synchronous, batch in → batch out                       |
| **Online**  | `ContinuousBatchingScheduler.step()` | Iteration-level scheduling, but no HTTP layer — only called from benchmarks |

Day 17 connects the online path to the outside world. After today, users can `POST` a prompt and get generated text back over HTTP.

---

## The 4 Files

### 1. RequestHandler (`serving/request_handler.py`) — Parse & Validate

**Role:** Convert raw HTTP JSON into a scheduler `Request` object.

**What it does:**

- Receives the JSON body from FastAPI (prompt, max_tokens)
- Validates inputs (prompt is non-empty string, max_tokens is positive integer within bounds)
- Tokenizes the prompt using the `Tokenizer`
- Creates a `Request` object (defined in `scheduler/request.py` — has `prompt`, `token_ids`, `max_tokens`, `state` lifecycle)
- Generates a unique `request_id` via `uuid.uuid4()`
- Returns the `Request` to the router

**Dependencies:** `Tokenizer` only (injected at construction).

**Interface:**

- `__init__(self, tokenizer, max_model_len)` — stores tokenizer and max context length for validation
- `handle(self, prompt, max_tokens) → Request` — validates, tokenizes, creates `Request`

**What it does NOT do:**

- Does not schedule anything — that's the router's job
- Does not run inference — that's the engine's job
- Does not know about HTTP/FastAPI — receives plain Python arguments
- Does not manage state — stateless, creates a new `Request` and hands it off

**Why separate from the router?** Separation of concerns. The handler is pure input parsing — easy to unit test (pass a string, check you get a valid `Request` back, check bad inputs raise errors). The router has complex orchestration logic. Mixing them makes both harder to test.

---

### 2. Router (`serving/router.py`) — Orchestrate

**Role:** The bridge between HTTP and the scheduler. Submits requests and runs the generation loop.

**What it does:**

- Holds references to: `ContinuousBatchingScheduler`, `RequestHandler`, `InferenceEngine` (or model + tokenizer)
- On incoming request: calls `request_handler.handle()` → gets `Request` → calls `scheduler.add_request()`
- Runs the **generation loop**: while `scheduler.has_work()`, call `scheduler.step()` which runs one decode iteration for all active requests
- After each step, checks which requests are finished (hit EOS or max_tokens)
- Collects completed results and returns them

**Key decisions:**

- Synchronous first — the generation loop blocks until the request is done. Simplest correct implementation.
- The router manages the lifecycle: submit → poll → return. The scheduler manages which requests run in each step.

---

### 3. API Server (`serving/api_server.py`) — FastAPI Application

**Role:** The HTTP entry point. Defines endpoints, initializes all components on startup.

**What it does:**

- On startup (lifespan): loads model, tokenizer, creates scheduler, creates router — all heavy initialization happens once
- Exposes `POST /generate` endpoint that accepts `{"prompt": "...", "max_tokens": 50}`
- Calls `router.generate()` → returns `{"request_id": "...", "generated_text": "...", "token_count": N}`
- Exposes `GET /health` for sanity checking

**Key decisions:**

- Model is loaded once at startup, not per-request
- Config comes from `configs/server_config.yaml` (host, port, timeout, max_concurrent_requests) and `configs/model_config.yaml` (model hyperparameters)

---

### 4. Client (`serving/client.py`) — Test Client

**Role:** Simple Python script that sends requests to the running server.

**What it does:**

- Sends `POST /generate` with a prompt and max_tokens
- Prints the response (generated text, token count, latency)
- Can send multiple requests sequentially for basic testing

Uses `httpx` (already in pyproject.toml dependencies).

---

## Request Lifecycle (End-to-End)

```bash
1. Client sends POST /generate {"prompt": "The meaning of life is", "max_tokens": 50}
2. api_server receives it, calls router.generate(prompt, max_tokens)
3. router calls request_handler.handle(prompt, max_tokens) → Request object
4. router calls scheduler.add_request(request)
5. router runs generation loop:
   while scheduler.has_work():
       scheduler.step()  ← one decode step for all active requests
6. Request finishes (EOS or max_tokens reached)
7. router detokenizes output, returns to api_server
8. api_server returns JSON response to client
```

## Dependency Flow

```bash
api_server.py
  └── router.py
        ├── request_handler.py
        │     └── Tokenizer, Request
        └── ContinuousBatchingScheduler
              └── InferenceEngine (model, tokenizer, cache)
```

## How Client and API Server Relate

The client and api_server are **separate processes** that communicate over HTTP. They never import each other.

```bash
api_server.py (Server — GPU machine)          client.py (Client — any machine)
─────────────────────────────────────          ─────────────────────────────────
Listens on port 8000                           Connects to http://localhost:8000
Receives HTTP requests                         Sends HTTP requests
Calls router.generate() internally             Calls httpx.post() externally
Returns JSON over HTTP                         Receives JSON over HTTP
Knows about Router, Scheduler, Engine          Knows only a URL
```

| Question           | api_server.py     | client.py            |
|--------------------|-------------------|----------------------|
| Which side?        | Server (receives) | Client (sends)       |
| Framework          | FastAPI           | httpx                |
| Needs GPU?         | Yes (runs model)  | No (just HTTP calls) |
| Knows about model? | Yes (via Router)  | No (just sends text) |
| Where it runs      | GPU machine       | Anywhere             |

## What Already Exists (Building On)

| Component                     | File                               | What It Provides                                                                           |
|-------------------------------|------------------------------------|--------------------------------------------------------------------------------------------|
| `Request`                     | `scheduler/request.py`             | `RequestState` (PENDING/RUNNING/FINISHED/FAILED), `add_generated_token()`, `is_finished()` |
| `RequestQueue`                | `scheduler/request_queue.py`       | FIFO queue with `add()`, `pop()`, `get_batch()`, `is_empty()`                              |
| `ContinuousBatchingScheduler` | `scheduler/continuous_batching.py` | `add_request()`, `step()`, `has_work()`, `complete_request()`                              |
| `InferenceEngine`             | `inference/inference_engine.py`    | `generate()` with `cache_type="standard"` and `cache_type="paged"`                         |
| `Tokenizer`                   | `tokenizer/tokenizer.py`           | `encode()`, `decode()`                                                                     |
| Server config                 | `configs/server_config.yaml`       | host, port, timeout, max_concurrent_requests                                               |

## Tests

- `test_request_handler.py`: Valid/invalid inputs, correct `Request` creation, edge cases (empty prompt, max_tokens=0)
- `test_router.py`: Submit request → get result, multiple requests, finished state check
- End-to-end: Start server → client sends request → verify response

## What Day 17 Does NOT Include

- **Streaming responses** (SSE/WebSocket) — synchronous only
- **Concurrent async handling** — one-at-a-time through the generation loop
- **Load testing** — that's Day 18
- **Authentication/rate limiting** — not in scope
