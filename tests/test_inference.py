import logging, torch, pytest

from llm_engine import Tokenizer, InferenceEngine
from llm_engine import load_asset_paths, load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def test_smoke_full_inference_path():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    max_tokens = 50
    sampling_method = 'greedy'
    engine = InferenceEngine(model = model, 
                             device = device, 
                             tokenizer = tokenizer,
                             eos_token_id = model_cfg['eos_token_id'],
                             sampling_method = sampling_method
                )
    
    # Test with both string and list inputs
    input_sentence = [
                        "Hello world, what is the meaning of life?",
                        ['Hello world, what is the capital of france'],
            ]
    for inp in input_sentence:
        output = engine.generate(inp, max_tokens = max_tokens)

        # Basic sanity checks on the output structure and content
        if isinstance(inp, str):
            assert isinstance(output, dict)
            assert output['input_text'] == inp
            assert output['sampling_method'] == sampling_method
            assert isinstance(output["token_count"], int) and output["token_count"] > 0
            assert isinstance(output["stop_reason"], str)
            assert isinstance(output["generated_text"], str) and len(output["generated_text"]) > 0
        elif isinstance(inp, list):
            assert isinstance(output, list)
            assert len(output) == len(inp)
            for i, res in enumerate(output):
                assert isinstance(res, dict)
                assert res['input_text'] == inp[i]
                assert res['sampling_method'] == sampling_method
                assert isinstance(res["token_count"], int) and res["token_count"] > 0
                assert isinstance(res["stop_reason"], str)
                assert isinstance(res["generated_text"], str) and len(res["generated_text"]) > 0

def test_correctness_with_diverse_prompts():
    
    config, model_cfg = load_asset_paths()
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'  # Force CPU for correctness testing to avoid GPU-related non-determinism
    engine = InferenceEngine(model = model, 
                             device = device, 
                             tokenizer = tokenizer,
                             eos_token_id = model_cfg['eos_token_id'])
    
    # Factual spot-checks
    factual_prompts = [
        ("The theory of relativity was developed by", "Einstein"),
        ("Water is made of hydrogen and", "oxygen"),
        ("The president of the United States is the head of", "United"),
        ("The capital of France is Paris, and the capital of Germany is", "Berlin"),
    ]
    for prompt, expected_word in factual_prompts:
        output = engine.generate(prompt, max_tokens = 50)
        text = output["generated_text"]
        if isinstance(text, list):
            text = text[0]
        # logger.info(f"Input: {prompt}, Output: {text}")
        assert expected_word in text, f"Expected '{expected_word}' in output for prompt: '{prompt}'"
    
    # Determinism check — same prompt, same output
    out1 = engine.generate("Once upon a time")
    out2 = engine.generate("Once upon a time")
    assert out1["generated_text"] == out2["generated_text"], "Greedy decoding should be deterministic"
    
    # Coherence — output starts with input
    prompt = "The weather today is"
    output = engine.generate(prompt)
    text = output["generated_text"]
    if isinstance(text, list):
        text = text[0]
    assert text.startswith(prompt), "Generated text should start with the input prompt"
    
def test_standard_kv_cache_smoke():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    max_tokens = 50
    sampling_method = 'greedy'
    engine = InferenceEngine(model = model, 
                             device = device, 
                             tokenizer = tokenizer,
                             eos_token_id = model_cfg['eos_token_id'],
                             sampling_method = sampling_method,
                             is_kv_cache_enabled = True,
                             max_tokens_for_kv_cache = model_cfg['n_ctx'],
                             batch_size = 1,
                             model_cfg = model_cfg         
                )
    
    # Test with both string and list inputs
    input_sentence = [
                        "Hello world, what is the meaning of life?",
                        ['Hello world, what is the capital of france'],
            ]
    for inp in input_sentence:
        output = engine.generate(inp, max_tokens = max_tokens)

        # Basic sanity checks on the output structure and content
        if isinstance(inp, str):
            assert isinstance(output, dict)
            assert output['input_text'] == inp
            assert output['sampling_method'] == sampling_method
            assert isinstance(output["token_count"], int) and output["token_count"] > 0
            assert isinstance(output["stop_reason"], str)
            assert isinstance(output["generated_text"], str) and len(output["generated_text"]) > 0
        elif isinstance(inp, list):
            assert isinstance(output, list)
            assert len(output) == len(inp)
            for i, res in enumerate(output):
                assert isinstance(res, dict)
                assert res['input_text'] == inp[i]
                assert res['sampling_method'] == sampling_method
                assert isinstance(res["token_count"], int) and res["token_count"] > 0
                assert isinstance(res["stop_reason"], str)
                assert isinstance(res["generated_text"], str) and len(res["generated_text"]) > 0

def test_standard_kv_cache_matches_no_cache():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    max_tokens = 50
    sampling_method = 'greedy'
    
    input_sentence = "Hello world, what is the meaning of life?"
    
    engine_no_cache = InferenceEngine(model = model, 
                                    device = device, 
                                    tokenizer = tokenizer,
                                    eos_token_id = model_cfg['eos_token_id'],
                                    sampling_method = sampling_method,
                )
    
    engine_cache = InferenceEngine(model = model, 
                                device = device, 
                                tokenizer = tokenizer,
                                eos_token_id = model_cfg['eos_token_id'],
                                sampling_method = sampling_method,
                                is_kv_cache_enabled = True,
                                max_tokens_for_kv_cache = model_cfg['n_ctx'],
                                batch_size = 1,
                                model_cfg = model_cfg         
                )
    
    output_no_cache = engine_no_cache.generate(input_sentence, max_tokens = max_tokens)
    output_cache = engine_cache.generate(input_sentence, max_tokens = max_tokens)
    
    assert output_no_cache["generated_text"] == output_cache["generated_text"], "Outputs should match between cache and no-cache modes for the same input and settings"
        
def test_standard_kv_cache_determinism():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    max_tokens = 50
    sampling_method = 'greedy'
    
    input_sentence = "Hello world, what is the meaning of life?"
    
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                            sampling_method = sampling_method,
                            is_kv_cache_enabled = True,
                            max_tokens_for_kv_cache = model_cfg['n_ctx'],
                            batch_size = 1,
                            model_cfg = model_cfg         
                )
    
    output1 = engine.generate(input_sentence, max_tokens = max_tokens)
    output2 = engine.generate(input_sentence, max_tokens = max_tokens)
    
    assert output1["generated_text"] == output2["generated_text"], "Greedy decoding with KV cache should be deterministic"
    
def test_standard_paged_kv_cache_smoke():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    max_tokens = 50
    num_blocks = 128
    block_size = 16
    sampling_method = 'greedy'
    
    engine = InferenceEngine(model = model, 
                             device = device, 
                             tokenizer = tokenizer,
                             eos_token_id = model_cfg['eos_token_id'],
                             sampling_method = sampling_method,
                             is_kv_cache_enabled = True,
                             max_tokens_for_kv_cache = model_cfg['n_ctx'],
                             batch_size = 1,
                             model_cfg = model_cfg,
                             cache_type = 'paged',
                             num_blocks = num_blocks,
                             block_size = block_size, 
                )
    
    # Test with both string and list inputs
    input_sentence = [
                        "Hello world, what is the meaning of life?",
                        ['Hello world, what is the capital of france'],
            ]
    for inp in input_sentence:
        output = engine.generate(inp, max_tokens = max_tokens)

        # Basic sanity checks on the output structure and content
        if isinstance(inp, str):
            assert isinstance(output, dict)
            assert output['input_text'] == inp
            assert output['sampling_method'] == sampling_method
            assert isinstance(output["token_count"], int) and output["token_count"] > 0
            assert isinstance(output["stop_reason"], str)
            assert isinstance(output["generated_text"], str) and len(output["generated_text"]) > 0
        elif isinstance(inp, list):
            assert isinstance(output, list)
            assert len(output) == len(inp)
            for i, res in enumerate(output):
                assert isinstance(res, dict)
                assert res['input_text'] == inp[i]
                assert res['sampling_method'] == sampling_method
                assert isinstance(res["token_count"], int) and res["token_count"] > 0
                assert isinstance(res["stop_reason"], str)
                assert isinstance(res["generated_text"], str) and len(res["generated_text"]) > 0

def test_paged_kv_cache_matches_standard():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    max_tokens = 50
    num_blocks = 128
    block_size = 16
    sampling_method = 'greedy'
    
    input_sentence = "Hello world, what is the meaning of life?"
    
    engine_cache = InferenceEngine(model = model, 
                                device = device, 
                                tokenizer = tokenizer,
                                eos_token_id = model_cfg['eos_token_id'],
                                sampling_method = sampling_method,
                                is_kv_cache_enabled = True,
                                max_tokens_for_kv_cache = model_cfg['n_ctx'],
                                batch_size = 1,
                                model_cfg = model_cfg         
                )
    
    engine_paged_cache = InferenceEngine(model = model, 
                                        device = device, 
                                        tokenizer = tokenizer,
                                        eos_token_id = model_cfg['eos_token_id'],
                                        sampling_method = sampling_method,
                                        is_kv_cache_enabled = True,
                                        max_tokens_for_kv_cache = model_cfg['n_ctx'],
                                        batch_size = 1,
                                        model_cfg = model_cfg,
                                        cache_type = 'paged',
                                        num_blocks = num_blocks,
                                        block_size = block_size, 
                )
    
    output_cache = engine_cache.generate(input_sentence, max_tokens = max_tokens)
    output_paged_cache = engine_paged_cache.generate(input_sentence, max_tokens = max_tokens)
    
    assert output_cache["generated_text"] == output_paged_cache["generated_text"], "Outputs should match between standard KV cache and paged KV cache modes for the same input and settings"
    
def test_paged_kv_cache_cleanup():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    max_tokens = 50
    num_blocks = 128
    block_size = 16
    sampling_method = 'greedy'
    
    input_sentence = "Hello world, what is the meaning of life?"
    
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                            sampling_method = sampling_method,
                            is_kv_cache_enabled = True,
                            max_tokens_for_kv_cache = model_cfg['n_ctx'],
                            batch_size = 1,
                            model_cfg = model_cfg,
                            cache_type = 'paged',
                            num_blocks = num_blocks,
                            block_size = block_size, 
                )
        
    output = engine.generate(input_sentence, max_tokens = max_tokens)
    
    assert len(engine.allocator.free_blocks) == num_blocks, "All blocks should be freed after generation"
    
def test_paged_kv_cache_sequential_calls():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    max_tokens = 50
    num_blocks = 128
    block_size = 16
    sampling_method = 'greedy'
    
    input_sentence = "Hello world, what is the meaning of life?"
    
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                            sampling_method = sampling_method,
                            is_kv_cache_enabled = True,
                            max_tokens_for_kv_cache = model_cfg['n_ctx'],
                            batch_size = 1,
                            model_cfg = model_cfg,
                            cache_type = 'paged',
                            num_blocks = num_blocks,
                            block_size = block_size, 
                )
    
    output1 = engine.generate(input_sentence, max_tokens = max_tokens)
    output2 = engine.generate(input_sentence, max_tokens = max_tokens)
    
    assert output1["generated_text"] == output2["generated_text"], "Outputs should match across sequential calls with paged KV cache enabled"
    
def test_batch_multi_prompt():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    max_tokens = 50
    num_blocks = 128
    block_size = 16
    sampling_method = 'greedy'
    
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                            sampling_method = sampling_method,
                            is_kv_cache_enabled = True,
                            max_tokens_for_kv_cache = model_cfg['n_ctx'],
                            batch_size = 1,
                            model_cfg = model_cfg,
                            cache_type = 'paged',
                            num_blocks = num_blocks,
                            block_size = block_size, 
                )
    
    # Pass 3+ prompts as a list (varying lengths).
    prompts = [
        "What is the capital of France?",
        "Who won the World Series in 2020?",
        "Explain the theory of relativity in simple terms.",
    ]
    
    outputs = engine.generate(prompts, max_tokens = max_tokens)
    
    assert isinstance(outputs, list) and len(outputs) == len(prompts), "Output should be a list of the same length as the input prompts"
    for i, output in enumerate(outputs):
        assert isinstance(output, dict), f"Each output should be a dictionary, but got {type(output)} for prompt: '{prompts[i]}'"
        assert 'input_text' in output and output['input_text'] == prompts[i], f"Output should contain the original prompt text for prompt: '{prompts[i]}'"
        assert 'generated_text' in output and isinstance(output['generated_text'], str) and len(output['generated_text']) > 0, f"Generated text should be a non-empty string for prompt: '{prompts[i]}'"
        assert 'sampling_method' in output and output['sampling_method'] == sampling_method, f"Output should contain the correct sampling method for prompt: '{prompts[i]}'"
        assert 'token_count' in output and isinstance(output['token_count'], int) and output['token_count'] > 0, f"Output should contain a valid token count for prompt: '{prompts[i]}'"
        assert 'stop_reason' in output and isinstance(output['stop_reason'], str), f"Output should contain a stop reason string for prompt: '{prompts[i]}'"
        
def test_batch_output_starts_with_prompt():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    max_tokens = 50
    num_blocks = 128
    block_size = 16
    sampling_method = 'greedy'
    
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                            sampling_method = sampling_method,
                            is_kv_cache_enabled = True,
                            max_tokens_for_kv_cache = model_cfg['n_ctx'],
                            batch_size = 1,
                            model_cfg = model_cfg,
                            cache_type = 'paged',
                            num_blocks = num_blocks,
                            block_size = block_size, 
                )
    
    prompts = [
        "What is the capital of France?",
        "Who won the World Series in 2020?",
        "Explain the theory of relativity in simple terms.",
    ]
    
    outputs = engine.generate(prompts, max_tokens = max_tokens)
    
    for i, output in enumerate(outputs):
        generated_text = output['generated_text']
        if isinstance(generated_text, list):
            generated_text = generated_text[0]
        assert generated_text.startswith(prompts[i]), f"Generated text should start with the original prompt for prompt: '{prompts[i]}'"
        
def test_batch_token_counts_independent():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    max_tokens = 50
    num_blocks = 128
    block_size = 16
    sampling_method = 'greedy'
    
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                            sampling_method = sampling_method,
                            is_kv_cache_enabled = True,
                            max_tokens_for_kv_cache = model_cfg['n_ctx'],
                            batch_size = 1,
                            model_cfg = model_cfg,
                            cache_type = 'paged',
                            num_blocks = num_blocks,
                            block_size = block_size, 
                )
    
    # Use prompts of very different lengths (e.g., 3-word vs 20-word).
    prompts = [
        "What is AI?",
        "In the context of machine learning, explain the bias-variance tradeoff and how it impacts model performance on unseen data.",
        "List the first 10 prime numbers."
    ]   
    
    output = engine.generate(prompts, max_tokens = max_tokens)
    
    for i, res in enumerate(output):
        assert 'token_count' in res and isinstance(res['token_count'], int) and res['token_count'] > 0, f"Output should contain a valid token count for prompt: '{prompts[i]}'"

def test_empty_string_raises():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                )
    
    with pytest.raises(ValueError, match = "Input prompt cannot be empty."):
        engine.generate("", 50)
        
def test_whitespace_only_raises():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                )
    
    with pytest.raises(ValueError, match = "Input prompt cannot be empty or whitespace-only."):
        engine.generate("   ", 50)
        
def test_empty_list_raises():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                )
    
    with pytest.raises(ValueError, match = "Input prompt list cannot be empty."):
        engine.generate([], 50)
        
def test_list_with_non_string_raises():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                )
    
    with pytest.raises(ValueError, match = "All items in the input prompt list must be strings."):
        engine.generate(["valid", 123], 50)
        
def test_init_none_tokenizer_raises():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    
    device = 'cpu'
    
    with pytest.raises(TypeError, match =  "tokenizer cannot be None"):
        InferenceEngine(model = model, 
                        device = device, 
                        tokenizer = None,
                        eos_token_id = model_cfg['eos_token_id'],
                )
        
def test_init_bad_model_raises():
    
    config, model_cfg = load_asset_paths()
    
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    
    with pytest.raises(TypeError, match = "Model must be an instance of the model class."):
        InferenceEngine(model = "not_a_model", 
                        device = device, 
                        tokenizer = tokenizer,
                        eos_token_id = model_cfg['eos_token_id'],
                )
        
def test_init_kv_cache_without_model_cfg_raises():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    
    with pytest.raises(ValueError, match = "model_cfg is required when is_kv_cache_enabled=True"):
        InferenceEngine(model = model, 
                        device = device, 
                        tokenizer = tokenizer,
                        eos_token_id = model_cfg['eos_token_id'],
                        is_kv_cache_enabled = True,
                        max_tokens_for_kv_cache = 512,
                        batch_size = 1,
                )
        
def test_init_invalid_cache_type_raises():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    cache_type = 'invalid_cache_type'
    
    with pytest.raises(ValueError, match = f"Unsupported cache_type '{cache_type}'. Supported types: 'standard', 'paged'."):
        InferenceEngine(model = model, 
                        device = device, 
                        tokenizer = tokenizer,
                        eos_token_id = model_cfg['eos_token_id'],
                        is_kv_cache_enabled = True,
                        max_tokens_for_kv_cache = 512,
                        batch_size = 1,
                        model_cfg = model_cfg,
                        cache_type = cache_type,
                )
        
def test_max_tokens_one():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                )
    
    output = engine.generate("Hello world", max_tokens = 1)
    
    assert isinstance(output, dict), "Output should be a dictionary"
    assert output['token_count'] == 1, "Token count should be 1 when max_tokens is set to 1"
    assert 'max_tokens' in output['stop_reason'], "Output should contain the max_tokens value"

def test_stop_reason_max_tokens():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                )
    
    output = engine.generate("Hello world", max_tokens = 5)
    
    assert isinstance(output, dict), "Output should be a dictionary"
    assert output['token_count'] == 5, "Token count should be 1 when max_tokens is set to 1"
    assert 'max_tokens' in output['stop_reason'], "Output should contain the max_tokens value"
    
def test_single_input_returns_string_not_list():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                )
    
    output = engine.generate("Hello world", max_tokens = 5)
    
    assert isinstance(output, dict), "Output should be a dictionary for single input prompt"
    assert isinstance(output['generated_text'], str), "Generated text should be a string for single input prompt"
    assert isinstance(output['stop_reason'], str), "Stop reason should be a string for single input prompt"
    assert isinstance(output['token_count'], int), "Token count should be an integer for single input prompt"
    
def test_list_input_returns_list_of_dicts():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cpu'
    engine = InferenceEngine(model = model, 
                            device = device, 
                            tokenizer = tokenizer,
                            eos_token_id = model_cfg['eos_token_id'],
                )
    
    prompts = ["Hello world", "What is AI?"]
    output = engine.generate(prompts, max_tokens = 5)
    
    assert isinstance(output, list), "Output should be a list for multiple input prompts"
    assert len(output) == len(prompts), "Output list length should match number of input prompts"
    for i, res in enumerate(output):
        assert isinstance(res, dict), f"Each item in output list should be a dictionary, but got {type(res)} for prompt: '{prompts[i]}'"
        assert 'generated_text' in res and isinstance(res['generated_text'], str), f"Generated text should be a string in output dictionary for prompt: '{prompts[i]}'"
        assert 'stop_reason' in res and isinstance(res['stop_reason'], str), f"Stop reason should be a string in output dictionary for prompt: '{prompts[i]}'"
        assert 'token_count' in res and isinstance(res['token_count'], int), f"Token count should be an integer in output dictionary for prompt: '{prompts[i]}'"