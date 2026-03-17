import logging, torch # type: ignore
logger = logging.getLogger(__name__)

from llm_engine import Tokenizer, InferenceEngine
from llm_engine import load_asset_paths, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

def test_smoke_full_inference_path():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    max_tokens = 500
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
        logger.info(f"Output: {output}")
        # Basic sanity checks on the output structure and content
        if isinstance(inp, str):
            assert isinstance(output, dict)
            assert output['input_text'] == inp
            assert output['sampling_method'] == sampling_method
            assert isinstance(output["token_count"], int) and output["token_count"] > 0
            assert isinstance(output["stop_reason"], str)
            # assert output["stop_reason"] in [f"Reached max_tokens limit of {max_tokens}.", "All sequences generated EOS token."]
        elif isinstance(inp, list):
            assert isinstance(output, list)
            assert len(output) == len(inp)
            for i, res in enumerate(output):
                assert isinstance(res, dict)
                assert res['input_text'] == inp[i]
                assert res['sampling_method'] == sampling_method
                assert isinstance(res["token_count"], int) and res["token_count"] > 0
                assert isinstance(res["stop_reason"], str)

def test_correctness_with_diverse_prompts():
    
    config, model_cfg = load_asset_paths()
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
        logger.info(f"Input: {prompt}, Output: {text}")
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