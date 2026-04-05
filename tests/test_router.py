import pytest, uuid, pytest_asyncio, asyncio, logging
from unittest.mock import MagicMock

from llm_engine import InferenceEngine, ContinuousBatchingScheduler, \
                       Router, RequestHandler, Tokenizer
from llm_engine import load_asset_paths

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
@pytest.fixture
def router():
    
    config, _ = load_asset_paths()
    tokenizer = Tokenizer(config = config)
    
    request_handler = RequestHandler(tokenizer = tokenizer, max_model_len = 1024)
    
    scheduler = ContinuousBatchingScheduler(max_batch_size = 4)
    
                       
    engine = MagicMock(spec = InferenceEngine)
    
    engine.generate.return_value = {"generated_text": "mocked output", \
                                    "token_count": 10, \
                                    "stop_reason": "max_tokens"}
    
    return Router(request_handler = request_handler, 
                    scheduler = scheduler, 
                    engine = engine)
        
@pytest.mark.asyncio     # ← on every async test
async def test_generate_valid_request(router):
    
    result = await router.generate("Hello_world", 50)
    
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "request_id" in result, "Result should contain request_id"
    assert "prompt" in result, "Result should contain prompt"
    assert "generated_text" in result, "Result should contain generated_text"
    assert "token_count" in result, "Result should contain token_count"
    assert "stop_reason" in result, "Result should contain stop_reason"
    assert result["generated_text"] == "mocked output", "Generated text should match mocked output"
    assert result["token_count"] == 10, "Token count should match mocked value"
    assert result["stop_reason"] == "max_tokens", "Stop reason should match mocked value"
    
@pytest.mark.asyncio     # ← on every async test
async def test_generate_request_id_is_uuid(router):
    
    result = await router.generate("Hello_world", 50)
    
    request_id = result['request_id']
    
    assert isinstance(request_id, str), "Request ID should be a string"
    assert len(request_id) == 36, "Request ID should be 36 characters long"
    assert request_id.count('-') == 4, "Request ID should contain 4 hyphens"
    assert uuid.UUID(request_id), "Request ID should be a valid UUID"
    
@pytest.mark.asyncio     # ← on every async test
async def test_generate_returns_correct_prompt(router):
    
    prompt = "Hello_world"
    result = await router.generate(prompt, 50)
    
    assert result["prompt"] == prompt, "Returned prompt should match input prompt" 
    
@pytest.mark.asyncio     # ← on every async test
async def test_generate_returns_engine_output(router):
    
    result = await router.generate("Hello_world", 50)
    
    assert result["generated_text"] == "mocked output", "Generated text should match mocked output"
    assert result["token_count"] == 10, "Token count should match mocked value"
    assert result["stop_reason"] == "max_tokens", "Stop reason should match mocked value"
    
@pytest.mark.asyncio     # ← on every async test
async def test_generate_request_state_finished(router):
    
    result = await router.generate("Hello_world", 50)
    
    request_id = result["request_id"]
    assert request_id in router.scheduler.completed_requests, "Request ID should be in scheduler's request states"
    assert router.scheduler.completed_requests[request_id].state == "finished", "Request state should be 'finished'"
    
@pytest.mark.asyncio     # ← on every async test
async def test_generate_engine_called_correctly(router):

    result = await router.generate("Hello world", 50)
    router.engine.generate.assert_called_once_with(input_text="Hello world", max_tokens=50)

@pytest.mark.asyncio     # ← on every async test
async def test_generate_invalid_prompt(router):
    
    with pytest.raises(ValueError, match = "prompt must be a non-empty string"):
        await router.generate("", 50)
        
@pytest.mark.asyncio     # ← on every async test
async def test_generate_invalid_max_tokens(router):
    
    with pytest.raises(ValueError, match = "max_tokens must be a positive integer"):
        await router.generate("Hello world", -10)
        
def test_constructor_rejects_none_handler():
    
    with pytest.raises(ValueError, match = "RequestHandler cannot be None"):
        Router(request_handler = None, scheduler = MagicMock(), engine = MagicMock())
        
def test_constructor_rejects_none_scheduler():
    
    with pytest.raises(ValueError, match = "Scheduler cannot be None"):
        Router(request_handler = MagicMock(), scheduler = None, engine = MagicMock())
        
def test_constructor_rejects_none_engine():
    
    with pytest.raises(ValueError, match = "InferenceEngine cannot be None"):
        Router(request_handler = MagicMock(), scheduler = MagicMock(), engine = None)
        
@pytest.mark.asyncio     # ← on every async test
async def test_multiple_sequential_generates(router):
    
    result1 = await router.generate(prompt = "First prompt", max_tokens = 20)
    result2 = await router.generate(prompt = "Second prompt", max_tokens = 30)
    
    assert result1['request_id'] != result2['request_id'], "Each generate call should produce a unique request ID"
    assert result1['prompt'] == "First prompt", "First result should contain the first prompt"
    assert result2['prompt'] == "Second prompt", "Second result should contain the second prompt"
    assert result1['generated_text'] == "mocked output", "First generated text should match mocked output"
    assert result2['generated_text'] == "mocked output", "Second generated text should match mocked output"
    assert result1['token_count'] == 10, "First token count should match mocked value"
    assert result2['token_count'] == 10, "Second token count should match mocked value"
    assert len(router.scheduler.completed_requests) == 2, "Scheduler should have two completed requests"
    assert router.scheduler.completed_requests[result1['request_id']].state == "finished", "First request state should be 'finished'"
    assert router.scheduler.completed_requests[result2['request_id']].state == "finished", "Second request state should be 'finished'"
    
@pytest_asyncio.fixture
async def async_router():
    
    config, _ = load_asset_paths()
    tokenizer = Tokenizer(config = config)
    
    request_handler = RequestHandler(tokenizer = tokenizer, max_model_len = 1024)
    
    scheduler = ContinuousBatchingScheduler(max_batch_size = 4)
         
    engine = MagicMock(spec = InferenceEngine)
    
    engine.generate.return_value = [{"generated_text": "mocked output",
                                    "token_count": 10,
                                    "stop_reason": "max_tokens"}]
    
    # Note: If any async test sends multiple concurrent requests that get batched together, 
    # the mock would need to return multiple dicts. You could use side_effect instead to 
    # dynamically return the right number of dicts based on the input:
    # engine.generate.side_effect = lambda input_text, **kwargs: [
    #                                                         {"generated_text": "mocked output", 
    #                                                          "token_count": 10, 
    #                                                          "stop_reason": "max_tokens"}
    #                                                         for _ in input_text
    #                                                     ]
    
    router =  Router(request_handler = request_handler, 
                    scheduler = scheduler, 
                    engine = engine,
                    async_mode = True)
    
    yield router
    router.stop()  # Ensure we stop the router's background tasks after each test
    
@pytest.mark.asyncio      # ← on every async test
async def test_async_generate_valid_request(async_router):
    
    async_router.start()  # Start the async router's background tasks
        
    result = await async_router.generate("Hello_world", 50)
    
    # logger.info(f"Async generate result: {result}")
    
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "request_id" in result, "Result should contain request_id"
    assert "prompt" in result, "Result should contain prompt"
    assert "generated_text" in result, "Result should contain generated_text"
    assert "token_count" in result, "Result should contain token_count"
    assert "stop_reason" in result, "Result should contain stop_reason"
    assert result["generated_text"] == "mocked output", "Generated text should match mocked output"
    assert result["token_count"] == 10, "Token count should match mocked value"
    assert result["stop_reason"] == "max_tokens", "Stop reason should match mocked value"
    
@pytest.mark.asyncio      # ← on every async test
async def test_async_generate_request_id_is_uuid(async_router):
    
    async_router.start()  # Start the async router's background tasks
    
    result = await async_router.generate("Hello_world", 50)
    
    request_id = result['request_id']
    
    assert isinstance(request_id, str), "Request ID should be a string"
    assert len(request_id) == 36, "Request ID should be 36 characters long"
    assert request_id.count('-') == 4, "Request ID should contain 4 hyphens"
    assert uuid.UUID(request_id), "Request ID should be a valid UUID"
    
@pytest.mark.asyncio     # ← on every async test
async def test_async_generate_returns_engine_output(async_router):
    
    async_router.start()  # Start the async router's background tasks
    
    result = await async_router.generate("Hello_world", 50)
    
    assert result["generated_text"] == "mocked output", "Generated text should match mocked output"
    assert result["token_count"] == 10, "Token count should match mocked value"
    assert result["stop_reason"] == "max_tokens", "Stop reason should match mocked value"
    

@pytest.mark.asyncio     # ← on every async test
async def test_async_generate_request_state_finished(async_router):
    
    async_router.start()  # Start the async router's background tasks
    
    result = await async_router.generate("Hello_world", 50)
    
    request_id = result["request_id"]
    assert request_id in async_router.scheduler.completed_requests, "Request ID should be in scheduler's request states"
    assert async_router.scheduler.completed_requests[request_id].state == "finished", "Request state should be 'finished'"
    
@pytest.mark.asyncio     # ← on every async test
async def test_async_generate_engine_called_correctly(async_router):
    
    async_router.start()  # Start the async router's background tasks
    
    result = await async_router.generate("Hello world", 50)
    async_router.engine.generate.assert_called_once_with(input_text=["Hello world"], max_tokens = [50])
    
@pytest.mark.asyncio     # ← on every async test
async def test_async_generate_invalid_prompt(async_router):
    
    with pytest.raises(ValueError, match = "prompt must be a non-empty string"):
        await async_router.generate("", 50)
        
@pytest.mark.asyncio     # ← on every async test
async def test_async_generate_invalid_max_tokens(async_router):
    
    with pytest.raises(ValueError, match = "max_tokens must be a positive integer"):
        await async_router.generate("Hello world", -10)
        
@pytest.mark.asyncio     # ← on every async test
async def test_async_multiple_concurrent_generates(async_router):
    
    async_router.start() # Start the async router's background tasks
    
    # coroutine means the function is not executed yet, it will be executed when we await it or run it in an event loop
    # By calling generate multiple times without awaiting, we create multiple pending tasks that can run concurrently 
    # when we await them together with asyncio.gather. This allows us to test the router's ability to handle multiple 
    # concurrent requests and ensure that it correctly generates unique request IDs, processes each request independently, 
    # and updates the scheduler's state for each completed request.
    task1 = async_router.generate("First prompt", 20)  # Returns coroutine (NOT started yet)
    task2 = async_router.generate("Second prompt", 30) # Returns coroutine (NOT started yet)
    task3 = async_router.generate("Third prompt", 40)  # Returns coroutine (NOT started yet)
    
    '''
    Description: 
        This test verifies that multiple concurrent generate calls produce unique request IDs, 
        return the correct prompts and engine outputs, and that the scheduler correctly tracks the completed requests. 
        We use asyncio.gather to run multiple generate calls concurrently and then assert that each result is correct 
        and that the scheduler has recorded all completed requests with the correct state.
    '''
    result1, result2, result3 = await asyncio.gather(task1, task2, task3)
    
    assert result1['request_id'] != result2['request_id'] != result3['request_id'], "Each generate call should produce a unique request ID"
    assert result1['prompt'] == "First prompt", "First result should contain the first prompt"
    assert result2['prompt'] == "Second prompt", "Second result should contain the second prompt"
    assert result3['prompt'] == "Third prompt", "Third result should contain the third prompt"
    assert result1['generated_text'] == "mocked output", "First generated text should match mocked output"
    assert result2['generated_text'] == "mocked output", "Second generated text should match mocked output"
    assert result3['generated_text'] == "mocked output", "Third generated text should match mocked output"
    assert result1['token_count'] == 10, "First token count should match mocked value"
    assert result2['token_count'] == 10, "Second token count should match mocked value"
    assert result3['token_count'] == 10, "Third token count should match mocked value"
    assert len(async_router.scheduler.completed_requests) == 3, "Scheduler should have three completed requests"
    assert async_router.scheduler.completed_requests[result1['request_id']].state == "finished", "First request state should be 'finished'"
    assert async_router.scheduler.completed_requests[result2['request_id']].state == "finished", "Second request state should be 'finished'"
    assert async_router.scheduler.completed_requests[result3['request_id']].state == "finished", "Third request state should be 'finished'"
