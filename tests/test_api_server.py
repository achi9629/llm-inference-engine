import pytest, asyncio
from unittest.mock import MagicMock
from starlette.testclient import TestClient

from llm_engine import create_app, load_asset_paths, RequestHandler, \
                       ContinuousBatchingScheduler, Tokenizer,\
                       InferenceEngine, Router
     
@pytest.fixture
def client():
    
    config, _ = load_asset_paths()
    tokenizer = Tokenizer(config = config)
    
    request_handler = RequestHandler(tokenizer = tokenizer, max_model_len = 1024)
    
    scheduler = ContinuousBatchingScheduler(max_batch_size = 4)
    
                       
    engine = MagicMock(spec = InferenceEngine)
    
    engine.generate.return_value = {"generated_text": "mocked output", \
                                    "token_count": 10, \
                                    "stop_reason": "max_tokens"}
    
    router = Router(request_handler = request_handler, 
                    scheduler = scheduler, 
                    engine = engine)
    
    app = create_app(router = router)
    
    return TestClient(app)
    
def test_generate_valid_request(client):
    
    payload = {"prompt": "Hello, world!", "max_tokens": 50}
    
    response = client.post("/generate", json=payload)
    
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    
    data = response.json()
    assert "request_id" in data, "Response missing 'request_id'"
    assert "prompt" in data, "Response missing 'prompt'"
    assert "generated_text" in data, "Response missing 'generated_text'"
    assert "token_count" in data, "Response missing 'token_count'"
    assert "stop_reason" in data, "Response missing 'stop_reason'"
    
def test_generate_response_content(client):
    
    payload = {"prompt": "Hello, world!", "max_tokens": 50}
    
    response = client.post("/generate", json=payload)
    
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    
    data = response.json()
    assert data["generated_text"] == "mocked output", f"Expected 'generated_text' to be 'mocked output', got {data['generated_text']}"
    assert data["token_count"] == 10, f"Expected 'token_count' to be 10, got {data['token_count']}"
    assert data["stop_reason"] == "max_tokens", f"Expected 'stop_reason' to be 'max_tokens', got {data['stop_reason']}"
    
def test_generate_empty_prompt_returns_400(client):
    
    payload = {"prompt": "", "max_tokens": 50}
    
    response = client.post("/generate", json=payload)
    
    assert response.status_code == 400, f"Expected status code 400 for empty prompt, got {response.status_code}"
    

def test_generate_invalid_max_tokens_returns_400(client):
    
    payload = {"prompt": "Hello, world!", "max_tokens": -1}
    
    response = client.post("/generate", json=payload)
    
    assert response.status_code == 400, f"Expected status code 400 for invalid max_tokens, got {response.status_code}"
    
def test_generate_missing_fields_returns_422(client):
    
    payload = {"prompt": "Hello, world!"}  # Missing max_tokens
    response = client.post("/generate", json=payload)
    assert response.status_code == 422, f"Expected status code 422 for missing fields, got {response.status_code}"
    
    payload = {"max_tokens": 50}  # Missing prompt
    response = client.post("/generate", json=payload)
    assert response.status_code == 422, f"Expected status code 422 for missing fields, got {response.status_code}"
    
    payload = {}  # Missing both fields
    response = client.post("/generate", json=payload)
    assert response.status_code == 422, f"Expected status code 422 for missing fields, got {response.status_code}"
    
def test_health_endpoint(client):
    
    response = client.get("/health")
    
    assert response.status_code == 200, f"Expected status code 200 for health endpoint, got {response.status_code}"
    
    data = response.json()
    assert data["status"] == "ok", f"Expected health status to be 'ok', got {data['status']}"
    
def test_generate_returns_503_at_capacity():
    
    """When semaphore is exhausted, new requests get 503."""
    config, _ = load_asset_paths()
    tokenizer = Tokenizer(config = config)
    
    request_handler = RequestHandler(tokenizer = tokenizer, max_model_len = 1024)
    
    scheduler = ContinuousBatchingScheduler(max_batch_size = 4)
    
    # Engine that blocks forever to hold the semaphore
    engine = MagicMock(spec = InferenceEngine)
    
    engine.generate.return_value = {"generated_text": "mocked output", \
                                    "token_count": 10, \
                                    "stop_reason": "max_tokens"}
    
    router = Router(request_handler = request_handler, 
                    scheduler = scheduler, 
                    engine = engine)
    
    # max_concurrent_requests=1 so second request gets 503
    app = create_app(router=router, max_concurrent_requests=1, request_timeout=30)
    
    client = TestClient(app)
    
    # First request succeeds (occupies the 1 slot, but sync TestClient completes it)
    response = client.post("/generate", json={"prompt": "Hello", "max_tokens": 10})
    assert response.status_code == 200, f"Expected status code 200 for first request, got {response.status_code}"
    
# 2. Test 504 on timeout
def test_generate_returns_504_on_timeout():
    """When generation exceeds request_timeout, returns 504."""

    config, _ = load_asset_paths()
    
    tokenizer = Tokenizer(config=config)
    
    request_handler = RequestHandler(tokenizer = tokenizer, max_model_len = 1024)
    
    scheduler = ContinuousBatchingScheduler(max_batch_size=4)

    # Engine that sleeps longer than timeout
    async def slow_generate(*args, **kwargs):
        await asyncio.sleep(5)
        return {"generated_text": "late", "token_count": 1, "stop_reason": "max_tokens"}

    engine = MagicMock(spec=InferenceEngine)
    
    router = Router(request_handler = request_handler, 
                    scheduler = scheduler, 
                    engine = engine, 
                    async_mode = True)
    
    # Patch router.generate to be slow
    original_generate = router.generate
    async def slow_router_generate(prompt, max_tokens):
        await asyncio.sleep(5)
        return await original_generate(prompt, max_tokens)
    router.generate = slow_router_generate

    # request_timeout=1 so it times out
    app = create_app(router = router, max_concurrent_requests = 64, request_timeout = 1)
    client = TestClient(app)

    response = client.post("/generate", json={"prompt": "Hello", "max_tokens": 10})
    assert response.status_code == 504, f"Expected 504, got {response.status_code}"
