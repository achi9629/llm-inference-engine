import torch, pytest # type: ignore

from llm_engine import load_asset_paths, load_model, generator, KVCache

batch_size = 1
config, model_cfg = load_asset_paths()

# customize model config for testing
model_cfg['n_layer'] = 2
model_cfg['n_heads'] = 2
model_cfg['n_embd'] = 32
model_cfg['n_ctx'] = 64
model_cfg['n_positions'] = 64
model_cfg['n_inner'] = 128
model_cfg['vocab_size'] = 100

model = load_model(config, model_cfg, is_weights = False)

@pytest.fixture
def kv_cache():
    return KVCache(batch_size = batch_size,
                   n_layers = model_cfg['n_layer'],
                   n_heads = model_cfg['n_heads'],
                   head_dim = model_cfg['n_embd'] // model_cfg['n_heads'],
                   max_seq_len = model_cfg['n_ctx'],
                   dtype = next(model.parameters()).dtype,
                   device = 'cpu',
            )

def test_output_shape_and_length_without_kv_cache():
    
    token_ids = torch.randint(0, model_cfg['vocab_size'], (batch_size, 5), dtype = torch.long)
    predicted_token_ids, token_count, _ = generator(model = model,
                                                    token_ids = token_ids,
                                                    device = 'cpu',
                                                    max_tokens = 3,
                                                    eos_token_id = model_cfg['eos_token_id'],
                                                    sampling_method = 'greedy',
                                                        )
    
    assert token_count <= 3, f"Expected token count to be at most 3, but got {token_count}"
    assert predicted_token_ids.shape[0] == batch_size, f"Expected batch size {batch_size}, but got {predicted_token_ids.shape[0]}"
    assert predicted_token_ids.shape[1] <= 5 + 3, f"Expected sequence length to be at most {5 + 3}, but got {predicted_token_ids.shape[1]}"
    
def test_output_shape_and_length_with_kv_cache(kv_cache):
    
    token_ids = torch.randint(0, model_cfg['vocab_size'], (batch_size, 5), dtype = torch.long)
    predicted_token_ids, token_count, _ = generator(model = model,
                                                    token_ids = token_ids,
                                                    device = 'cpu',
                                                    max_tokens = 3,
                                                    eos_token_id = model_cfg['eos_token_id'],
                                                    sampling_method = 'greedy',
                                                    kv_cache = kv_cache
                                                        )
    
    assert token_count <= 3, f"Expected token count to be at most 3, but got {token_count}"
    assert predicted_token_ids.shape[0] == batch_size, f"Expected batch size {batch_size}, but got {predicted_token_ids.shape[0]}"
    assert predicted_token_ids.shape[1] <= 5 + 3, f"Expected sequence length to be at most {5 + 3}, but got {predicted_token_ids.shape[1]}"


def test_cache_vs_no_cache_identical_tokens(kv_cache):
    
    model.eval()
    seed_value = 42
    
    torch.manual_seed(seed_value)
    token_ids_no_cache = torch.randint(0, model_cfg['vocab_size'], (batch_size, 5), dtype = torch.long)
    
    predicted_token_ids_no_cache, _, _ = generator(model = model,
                                                   token_ids = token_ids_no_cache,
                                                   device = 'cpu',
                                                   max_tokens = 3,
                                                   eos_token_id = model_cfg['eos_token_id'],
                                                   sampling_method = 'greedy',
                                            )
    
    torch.manual_seed(seed_value)
    token_ids_cache = torch.randint(0, model_cfg['vocab_size'], (batch_size, 5), dtype = torch.long)
    
    predicted_token_ids_cache, _, _ = generator(model = model,
                                                token_ids = token_ids_cache,
                                                device = 'cpu',
                                                max_tokens = 3,
                                                eos_token_id = model_cfg['eos_token_id'],
                                                sampling_method = 'greedy',
                                                kv_cache = kv_cache
                                            )
    
    assert predicted_token_ids_no_cache.equal(predicted_token_ids_cache), "Expected predicted token IDs to be identical with and without KV cache, but they differ"
    
def test_1d_input_returns_1d():
    
    token_ids = torch.randint(0, model_cfg['vocab_size'], (5,), dtype = torch.long)
    
    predicted_token_ids, _, _ = generator(model = model,
                                          token_ids = token_ids,
                                          device = 'cpu',
                                          max_tokens = 3,
                                          eos_token_id = model_cfg['eos_token_id'],
                                          sampling_method = 'greedy',
                                            )
    
    assert predicted_token_ids.dim() == 1, f"Expected output to be 1D, but got {predicted_token_ids.dim()}D"
    
def test_token_count_matches_generated():
    
    token_ids = torch.randint(0, model_cfg['vocab_size'], (batch_size, 5), dtype = torch.long)
    
    predicted_token_ids, token_count, _ = generator(model = model,
                                                    token_ids = token_ids,
                                                    device = 'cpu',
                                                    max_tokens = 3,
                                                    eos_token_id = model_cfg['eos_token_id'],
                                                    sampling_method = 'greedy',
                                            )
    
    assert token_count == predicted_token_ids.shape[1] - 5, f"Expected token count {token_count} to match the number of generated tokens {predicted_token_ids.shape[1] - 5}, but they differ"
    
def test_max_tokens_stop_reason():
    
    token_ids = torch.randint(0, model_cfg['vocab_size'], (batch_size, 5), dtype = torch.long)
    
    predicted_token_ids, token_count, stop_reason = generator(model = model,
                                                              token_ids = token_ids,
                                                              device = 'cpu',
                                                              max_tokens = 3,
                                                              eos_token_id = -1,  # Set to -1 to ensure EOS token is not generated
                                                              sampling_method = 'greedy',
                                                        )
    
    assert stop_reason[0] == f"Reached max_tokens limit of {3}.", f"Expected stop reason to indicate max_tokens limit, but got '{stop_reason}'"
    assert token_count == 3, f"Expected token count to be exactly 3 when stopping due to max_tokens, but got {token_count}"