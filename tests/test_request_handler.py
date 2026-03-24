import pytest, logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

from llm_engine import RequestHandler, Tokenizer, Request, load_asset_paths

config, model_cfg = load_asset_paths()
    
tokenizer = Tokenizer(config)
max_model_len = model_cfg['n_ctx']

@pytest.fixture
def handler():
    return RequestHandler(tokenizer = tokenizer, max_model_len = max_model_len)
    
def test_handle_valid_request(handler):
    
    request = handler.handle(prompt = "What is the capital of France?", max_tokens = 50)
    
    assert isinstance(request, Request), "The handler should return an instance of Request"
    assert request.prompt == "What is the capital of France?", "Prompt should be correctly set in the request"
    assert request.max_tokens == 50, "Max tokens should be correctly set in the request"
    assert len(request.token_ids) > 0, "Token IDs should be generated for the prompt"
    assert request.state == 'pending', "Request state should be initialized to 'pending'"
    assert request.generated_tokens == [], "Generated tokens should be initialized as an empty list"
    assert request.request_id.strip(), "Request ID should be generated and not be empty"
    
def test_handle_unique_request_ids(handler):
    
    request1 = handler.handle(prompt = "What is the capital of France?", max_tokens = 50)
    request2 = handler.handle(prompt = "What is the capital of France?", max_tokens = 50)
    
    assert request1.request_id != request2.request_id, "Each request should have a unique request ID"
    
def test_handle_empty_prompt(handler):
    
    with pytest.raises(ValueError, match = "prompt must be a non-empty string"):
        handler.handle(prompt = "", max_tokens = 50)
        
    with pytest.raises(ValueError, match = "prompt must be a non-empty string"):
        handler.handle(prompt = '', max_tokens = 50)

def test_handle_whitespace_prompt(handler):
    
    with pytest.raises(ValueError, match = "prompt must be a non-empty string"):
        handler.handle(prompt = "  ", max_tokens = 50)
        
    with pytest.raises(ValueError, match = "prompt must be a non-empty string"):
        handler.handle(prompt = '  ', max_tokens = 50)
        
def test_handle_non_string_prompt(handler):
    
    with pytest.raises(ValueError, match = "prompt must be a non-empty string"):
        handler.handle(prompt = 123, max_tokens = 50)
        
    with pytest.raises(ValueError, match = "prompt must be a non-empty string"):
        handler.handle(prompt = None, max_tokens = 50)
        
def test_handle_invalid_max_tokens_zero(handler):
    
    with pytest.raises(ValueError, match = "max_tokens must be a positive integer"):
        handler.handle(prompt = "What is the capital of France?", max_tokens = 0)
        
def test_handle_invalid_max_tokens_negative(handler):
    
    with pytest.raises(ValueError, match = "max_tokens must be a positive integer"):
        handler.handle(prompt = "What is the capital of France?", max_tokens = -10)
        
def test_handle_max_tokens_exceeds_model_len(handler):
    
    with pytest.raises(ValueError, match = f"max_tokens cannot exceed the model's maximum context length of {max_model_len}"):
        handler.handle(prompt = "What is the capital of France?", max_tokens = max_model_len + 1)
        
def test_constructor_invalid_tokenizer():
    
    with pytest.raises(ValueError, match = "tokenizer must be an instance of Tokenizer"):
        _ = RequestHandler(tokenizer = "not_a_tokenizer", max_model_len = max_model_len)
        
def test_handle_tokenization_correct(handler):
    
    request = handler.handle(prompt = "What is the capital of France?", max_tokens = 50)
    
    token_ids = tokenizer.encode("What is the capital of France?", return_tensor = False)[0]
    
    assert request.token_ids == token_ids, "The token IDs in the request should match the tokenizer output for the prompt"
    
def test_constructor_invalid_max_model_len():
    
    with pytest.raises(ValueError, match = "max_model_len must be a positive integer"):
        _ = RequestHandler(tokenizer = tokenizer, max_model_len = 0)
        
    with pytest.raises(ValueError, match = "max_model_len must be a positive integer"):
        _ = RequestHandler(tokenizer = tokenizer, max_model_len = -100)