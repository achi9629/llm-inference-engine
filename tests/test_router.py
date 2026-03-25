import pytest, uuid
from unittest.mock import MagicMock

from llm_engine import InferenceEngine, ContinuousBatchingScheduler, \
                       Router, RequestHandler, Tokenizer
from llm_engine import load_asset_paths
    
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
    


def test_generate_valid_request(router):
    
    result = router.generate("Hello_world", 50)
    
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "request_id" in result, "Result should contain request_id"
    assert "prompt" in result, "Result should contain prompt"
    assert "generated_text" in result, "Result should contain generated_text"
    assert "token_count" in result, "Result should contain token_count"
    assert "stop_reason" in result, "Result should contain stop_reason"
    assert result["generated_text"] == "mocked output", "Generated text should match mocked output"
    assert result["token_count"] == 10, "Token count should match mocked value"
    assert result["stop_reason"] == "max_tokens", "Stop reason should match mocked value"
    
def test_generate_request_id_is_uuid(router):
    
    result = router.generate("Hello_world", 50)
    
    request_id = result['request_id']
    
    assert isinstance(request_id, str), "Request ID should be a string"
    assert len(request_id) == 36, "Request ID should be 36 characters long"
    assert request_id.count('-') == 4, "Request ID should contain 4 hyphens"
    assert uuid.UUID(request_id), "Request ID should be a valid UUID"
    
def test_generate_returns_correct_prompt(router):
    
    prompt = "Hello_world"
    result = router.generate(prompt, 50)
    
    assert result["prompt"] == prompt, "Returned prompt should match input prompt" 
    
def test_generate_returns_engine_output(router):
    
    result = router.generate("Hello_world", 50)
    
    assert result["generated_text"] == "mocked output", "Generated text should match mocked output"
    assert result["token_count"] == 10, "Token count should match mocked value"
    assert result["stop_reason"] == "max_tokens", "Stop reason should match mocked value"
    
def test_generate_request_state_finished(router):
    
    result = router.generate("Hello_world", 50)
    
    request_id = result["request_id"]
    assert request_id in router.scheduler.completed_requests, "Request ID should be in scheduler's request states"
    assert router.scheduler.completed_requests[request_id].state == "finished", "Request state should be 'finished'"
    
def test_generate_engine_called_correctly(router):

    result = router.generate("Hello world", 50)
    router.engine.generate.assert_called_once_with(input_text="Hello world", max_tokens=50)

def test_generate_invalid_prompt(router):
    
    with pytest.raises(ValueError, match = "prompt must be a non-empty string"):
        router.generate("", 50)
        
def test_generate_invalid_max_tokens(router):
    
    with pytest.raises(ValueError, match = "max_tokens must be a positive integer"):
        router.generate("Hello world", -10)
        
def test_constructor_rejects_none_handler():
    
    with pytest.raises(ValueError, match = "RequestHandler cannot be None"):
        Router(request_handler = None, scheduler = MagicMock(), engine = MagicMock())
        
def test_constructor_rejects_none_scheduler():
    
    with pytest.raises(ValueError, match = "Scheduler cannot be None"):
        Router(request_handler = MagicMock(), scheduler = None, engine = MagicMock())
        
def test_constructor_rejects_none_engine():
    
    with pytest.raises(ValueError, match = "InferenceEngine cannot be None"):
        Router(request_handler = MagicMock(), scheduler = MagicMock(), engine = None)
        
def test_multiple_sequential_generates(router):
    
    result1 = router.generate(prompt = "First prompt", max_tokens = 20)
    result2 = router.generate(prompt = "Second prompt", max_tokens = 30)
    
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