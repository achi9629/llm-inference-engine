"""
Simple text generation example using the LLM Inference Engine.

Usage:
    PYTHONPATH=. python examples/simple_generation.py
"""

import torch # type: ignore

from llm_engine import Tokenizer, InferenceEngine, load_asset_paths, load_model

def main():
    
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
                             model_cfg = model_cfg,
                )
    
    prompts = [
        "The future of artificial intelligence",
        "In a galaxy far far away",
        "The key to understanding transformers is",
    ]
    
    for prompt in prompts:
        
        result = engine.generate(prompt, max_tokens)
        
        print('\nPrompt:', result ['input_text'])
        print('Generated Text:', result ['generated_text'][0])
        print('Sampling Method:', result ['sampling_method'])
        print('Tokens Generated:', result ['token_count'])
        print(f"Stop Reason: {result ['stop_reason']}")
        
if __name__ == "__main__":
    main()