import torch, pytest

from llm_engine import load_asset_paths, load_model, generator, KVCache

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

batch_size = 5
config, model_cfg = load_asset_paths()

# customize model config for testing
model_cfg['n_layer'] = 2
model_cfg['n_heads'] = 2
model_cfg['n_embd'] = 32
model_cfg['n_ctx'] = 64
model_cfg['n_positions'] = 64
model_cfg['n_inner'] = 128
model_cfg['vocab_size'] = 50257
eos_token = model_cfg['eos_token_id']
device = 'cpu'

model = load_model(config, model_cfg, is_weights = False)

@pytest.fixture
def bs_param(request):
    return request.param

@pytest.fixture
def kv_cache(bs_param: int):
    return KVCache(batch_size = bs_param,
                    n_layers = model_cfg['n_layer'],
                    n_heads = model_cfg['n_heads'],
                    head_dim = model_cfg['n_embd'] // model_cfg['n_heads'],
                    max_seq_len = model_cfg['n_ctx'],
                    dtype = next(model.parameters()).dtype,
                    device = device)

@pytest.mark.parametrize("bs_param", [5], indirect=True)
def test_output_shape(kv_cache):
    
    assert kv_cache.batch_size  == 5, f"Expected kv_cache batch size to be 5, but got {kv_cache.batch_size}"

    token_ids = torch.tensor([[eos_token, eos_token, 1, 2, 3], 
                          [eos_token, eos_token, eos_token, 6, 7],
                          [eos_token, eos_token, eos_token, eos_token, 11], 
                          [eos_token, 16, 17, 18, 19],
                          [99, 16, 17, 18, 19]], dtype=torch.long, device = device)

    padding_mask = torch.tensor([[0, 0, 1, 1, 1],
                                [0, 0, 0, 1, 1],
                                [0, 0, 0, 0, 1],
                                [0, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1]], dtype=torch.bool, device = device)
    
    output_token_ids, token_count, stop_reason = generator(model = model, 
                                                           token_ids = token_ids,
                                                           device = device,
                                                           max_tokens= 7, 
                                                           eos_token_id = eos_token,
                                                           padding_mask = padding_mask,
                                                           sampling_method = 'greedy',
                                                    )
    
    assert output_token_ids.shape[0] == batch_size, f"Expected output batch size {batch_size}, but got {output_token_ids.shape[0]}"
    assert output_token_ids.shape[1] <= 7 + 5, f"Expected output sequence length <= 12, but got {output_token_ids.shape[1]}"
    assert isinstance(token_count, torch.Tensor), f"Expected token_count to be a torch.Tensor, but got {type(token_count)}"
    assert token_count.shape == (batch_size,), f"Expected token_count shape (5,), but got {token_count.shape}"
    assert isinstance(stop_reason, list), f"Expected stop_reason to be a list, but got {type(stop_reason)}"
    assert len(stop_reason) == batch_size, f"Expected stop_reason length {batch_size}, but got {len(stop_reason)}"
    
@pytest.mark.parametrize("bs_param", [2], indirect=True)
def test_padding_doesnt_corrupt(kv_cache):
    
    assert kv_cache.batch_size  == 2, f"Expected kv_cache batch size to be 2, but got {kv_cache.batch_size }"
    
    token_ids_single = torch.tensor([1, 2, 3], dtype=torch.long, device = device) 

    token_ids_batch = torch.tensor([[eos_token, eos_token, 1, 2, 3],
                                    [eos_token, 6, 7, 8, 9]], dtype=torch.long, device = device)

    padding_mask =    torch.tensor([[0, 0, 1, 1, 1],
                                    [0, 1, 1, 1, 1]], dtype=torch.bool, device = device)
    
    output_token_ids_single, _, _ = generator(model = model, 
                                            token_ids = token_ids_single,
                                            device = device,
                                            max_tokens= 7, 
                                            eos_token_id = eos_token,
                                            padding_mask = None,
                                            sampling_method = 'greedy',
                                                    )

    output_token_ids_batch, _, _ = generator(model = model, 
                                            token_ids = token_ids_batch,
                                            device = device,
                                            max_tokens= 7, 
                                            eos_token_id = eos_token,
                                            padding_mask = padding_mask,
                                            sampling_method = 'greedy',
                                                    )
    kv_cache.reset_cache()
    output_token_ids_batch_cache, _, _ = generator(model = model, 
                                                    token_ids = token_ids_batch,
                                                    device = device,
                                                    max_tokens= 7, 
                                                    eos_token_id = eos_token,
                                                    padding_mask = padding_mask,
                                                    sampling_method = 'greedy',
                                                    kv_cache = kv_cache,
                                                    )
    
    n_pad = padding_mask[0].shape[0] -  padding_mask[0].sum().item()
    assert output_token_ids_single.equal(output_token_ids_batch[0,n_pad:]), f"Expected output token ids to be the same after padding, but got {output_token_ids_single} and {output_token_ids_batch[0,n_pad:]}"
    assert output_token_ids_single.equal(output_token_ids_batch_cache[0,n_pad:]), f"Expected output token ids to be the same after padding with kv_cache, but got {output_token_ids_single} and {output_token_ids_batch_cache[0,n_pad:]}"
    
@pytest.mark.parametrize("bs_param", [4], indirect=True)
def test_per_sequence_token_count(kv_cache):
    
    assert kv_cache.batch_size  == 4, f"Expected kv_cache batch size to be 4, but got {kv_cache.batch_size }"
    
    token_ids_batch = torch.tensor([[eos_token, eos_token, 1, 2, 3],
                                    [eos_token, 6, 7, 8, 9],
                                    [1, 2, 3, 4, 5],
                                    [eos_token, eos_token, eos_token, 80, 55]], dtype=torch.long, device = device)

    padding_mask =    torch.tensor([[0, 0, 1, 1, 1],
                                    [0, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 1],
                                    [0, 0, 0, 1, 1]], dtype=torch.bool, device = device)
    
    output_ids_no_cache, token_count_no_cache, _ = generator(model = model, 
                                            token_ids = token_ids_batch,
                                            device = device,
                                            max_tokens= 10, 
                                            eos_token_id = eos_token,
                                            padding_mask = padding_mask,
                                            sampling_method = 'greedy',
                        )
    
    kv_cache.reset_cache()
    output_ids_cache, token_count_cache, _ = generator(model = model, 
                                        token_ids = token_ids_batch,
                                        device = device,
                                        max_tokens= 10, 
                                        eos_token_id = eos_token,
                                        padding_mask = padding_mask,
                                        sampling_method = 'greedy',
                                        kv_cache = kv_cache,
                        )
    
    assert token_count_no_cache.shape == (kv_cache.batch_size,), f"Expected token_count shape {(kv_cache.batch_size,)}, but got {token_count_no_cache.shape}"
    assert token_count_cache.shape == (kv_cache.batch_size,), f"Expected token_count shape {(kv_cache.batch_size,)}, but got {token_count_cache.shape}"
    for i in range(kv_cache.batch_size):
        assert token_count_no_cache[i].item() <= 10, f"Expected token count for sequence {i} to be <= 10, but got {token_count_no_cache[i].item()}"
        assert token_count_cache[i].item() <= 10, f"Expected token count for sequence {i} to be <= 10, but got {token_count_cache[i].item()}"
        assert output_ids_no_cache[i].shape[0] - token_ids_batch[i].shape[0] == token_count_no_cache[i].item(), f"Expected output sequence length - input sequence length to equal token count for sequence {i} without cache, but got {output_ids_no_cache[i].shape[0] - token_ids_batch[i].shape[0]} and {token_count_no_cache[i].item()}"
        assert output_ids_cache[i].shape[0] - token_ids_batch[i].shape[0] == token_count_cache[i].item(), f"Expected output sequence length - input sequence length to equal token count for sequence {i} with cache, but got {output_ids_cache[i].shape[0] - token_ids_batch[i].shape[0]} and {token_count_cache[i].item()}"
        

def test_stop_reason_per_sequence():
    
    token_ids_batch = torch.tensor([[eos_token, eos_token, 1, 2, 3],
                                    [eos_token, 6, 7, 8, 9],
                                    [1, 2, 3, 4, 5],
                                    [eos_token, eos_token, eos_token, 80, 55]], dtype=torch.long, device = device)

    padding_mask =    torch.tensor([[0, 0, 1, 1, 1],
                                    [0, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 1],
                                    [0, 0, 0, 1, 1]], dtype=torch.bool, device = device)
    
    max_tokens = 3
    _, _, stop_reason = generator(model = model, 
                                token_ids = token_ids_batch,
                                device = device,
                                max_tokens= max_tokens, 
                                eos_token_id = eos_token,
                                padding_mask = padding_mask,
                                sampling_method = 'greedy',
                        )
    assert len(stop_reason) == 4, f"Expected stop_reason length to be 4, but got {len(stop_reason)}"
    for i in range(4):
        assert isinstance(stop_reason[i], str), f"Expected stop reason for sequence {i} to be a string, but got {type(stop_reason[i])}"
        assert stop_reason[i] != "", f"Expected stop reason for sequence {i} to be non-empty, but got an empty string"
        assert stop_reason[i] == f"Reached max_tokens limit of {max_tokens}.", f"Expected stop reason for sequence {i} to be 'Sequences generated EOS token.', but got {stop_reason[i]}"
        
@pytest.mark.parametrize("bs_param", [1], indirect=True)
def test_padding_mask_grows(kv_cache):
    
    assert kv_cache.batch_size == 1, f"Expected kv_cache batch size to be 1, but got {kv_cache.batch_size}"
    
    token_ids = torch.tensor([[eos_token, eos_token, 1, 2, 3]], dtype=torch.long, device = device) 
    padding_mask =    torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool, device = device)
    
    max_tokens = 4
    
    kv_cache.reset_cache()
    output_ids, token_count, stop_reason = generator(model = model, 
                                                        token_ids = token_ids,
                                                        device = device,
                                                        max_tokens= max_tokens, 
                                                        eos_token_id = eos_token,
                                                        padding_mask = padding_mask,
                                                        sampling_method = 'greedy',
                                                        kv_cache = kv_cache
                                                )

    assert output_ids.shape[1] > token_ids.shape[1], f"Expected output sequence length to be greater than input sequence length, but got {output_ids.shape[1]} and {token_ids.shape[0]}"
    assert output_ids[0][0] == eos_token and output_ids[0][1] == eos_token, f"Expected first two tokens of output to be EOS token, but got {output_ids[0][0]} and {output_ids[0][1]}"
    assert output_ids[:, :5].equal(token_ids), f"Expected first 5 tokens of output to match input token ids, but got {output_ids[:,:5]} and {token_ids}"
    
def test_left_padded_vs_no_pad():
    
    token_ids_single = torch.tensor([1, 1, 1], dtype=torch.long, device = device)
    token_ids_batch = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long, device = device)
    padding_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool, device = device)
    
    output_ids_single, _, _ = generator(model = model, 
                                        token_ids = token_ids_single,
                                        device = device,
                                        max_tokens= 5, 
                                        eos_token_id = eos_token,
                                        sampling_method = 'greedy',
                                )
    
    output_ids_batch, _, _ = generator(model = model, 
                                        token_ids = token_ids_batch,
                                        device = device,
                                        max_tokens= 5, 
                                        eos_token_id = eos_token,
                                        padding_mask = padding_mask,
                                        sampling_method = 'greedy',
                                )

    assert output_ids_batch[0,2:].equal(output_ids_single), f"Expected output token ids to be the same after padding, but got {output_ids_batch[0,2:]} and {output_ids_single}"
    
@pytest.mark.parametrize("bs_param", [4], indirect=True)
def test_all_same_length_no_padding_needed(kv_cache):
    
    assert kv_cache.batch_size == 4, f"Expected kv_cache batch size to be 4, but got {kv_cache.batch_size}"
    

    token_ids = torch.tensor([[3, 6, 1, 2, 3],
                            [4, 6, 7, 8, 9],
                            [1, 2, 3, 4, 5],
                            [7, 23, 23, 80, 55]], dtype=torch.long, device = device)

    padding_mask =    torch.tensor([[1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1]], dtype=torch.bool, device = device)
    
    
    output_ids_no_padding, _, _ = generator(model = model, 
                                        token_ids = token_ids,
                                        device = device,
                                        max_tokens= 5, 
                                        eos_token_id = eos_token,
                                        sampling_method = 'greedy',
                                )
    
    output_ids_padding, _, _ = generator(model = model, 
                                        token_ids = token_ids,
                                        device = device,
                                        max_tokens= 5, 
                                        eos_token_id = eos_token,
                                        padding_mask = padding_mask,
                                        sampling_method = 'greedy',
                                )
    
    kv_cache.reset_cache()
    output_ids_no_padding_cache, _, _ = generator(model = model, 
                                        token_ids = token_ids,
                                        device = device,
                                        max_tokens= 5, 
                                        eos_token_id = eos_token,
                                        sampling_method = 'greedy',
                                        kv_cache = kv_cache,
                                )
    
    kv_cache.reset_cache()
    output_ids_padding_cache, _, _ = generator(model = model, 
                                        token_ids = token_ids,
                                        device = device,
                                        max_tokens= 5, 
                                        eos_token_id = eos_token,
                                        padding_mask = padding_mask,
                                        sampling_method = 'greedy',
                                        kv_cache = kv_cache,
                                )
    
    assert output_ids_no_padding.equal(output_ids_padding), f"Expected output token ids to be the same with or without padding mask when no padding is needed, but got {output_ids_no_padding} and {output_ids_padding}"
    assert output_ids_no_padding.equal(output_ids_no_padding_cache), f"Expected output token ids to be the same with or without cache when no padding is needed, but got {output_ids_no_padding} and {output_ids_no_padding_cache}"
    assert output_ids_no_padding.equal(output_ids_padding_cache), f"Expected output token ids to be the same with or without padding mask and cache when no padding is needed, but got {output_ids_no_padding} and {output_ids_padding_cache}"
    
@pytest.mark.parametrize("bs_param", [1], indirect=True)
def test_single_item_batch(kv_cache):
    
    assert kv_cache.batch_size == 1, f"Expected kv_cache batch size to be 1, but got {kv_cache.batch_size}"
    
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device = device)
    padding_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool, device = device)
    

    output_token_ids, token_count, stop_reason = generator(model = model, 
                                                            token_ids = token_ids,
                                                            device = device,
                                                            max_tokens= 5, 
                                                            eos_token_id = eos_token,
                                                            padding_mask = padding_mask,
                                                            sampling_method = 'greedy',
                                                    )
    
    kv_cache.reset_cache()
    output_token_ids_cache, token_count_cache, stop_reason_cache = generator(model = model, 
                                                                            token_ids = token_ids,
                                                                            device = device,
                                                                            max_tokens= 5, 
                                                                            eos_token_id = eos_token,
                                                                            padding_mask = padding_mask,
                                                                            sampling_method = 'greedy',
                                                                            kv_cache = kv_cache,
                                                                    )
    
    assert output_token_ids.shape[0] == kv_cache.batch_size, f"Expected output batch size {kv_cache.batch_size}, but got {output_token_ids.shape[0]}"
    assert output_token_ids_cache.shape[0] == kv_cache.batch_size, f"Expected output batch size {kv_cache.batch_size}, but got {output_token_ids_cache.shape[0]}"
    assert output_token_ids.shape[1] <= 8, f"Expected output sequence length <= 8, but got {output_token_ids.shape[1]}"
    assert output_token_ids_cache.shape[1] <= 8, f"Expected output sequence length <= 8, but got {output_token_ids_cache.shape[1]}"
    assert token_count.shape == (kv_cache.batch_size,), f"Expected token_count shape {(kv_cache.batch_size,)}, but got {token_count.shape}"
    assert token_count_cache.shape == (kv_cache.batch_size,), f"Expected token_count shape {(kv_cache.batch_size,)}, but got {token_count_cache.shape}"
    assert len(stop_reason) == kv_cache.batch_size, f"Expected stop_reason length {kv_cache.batch_size}, but got {len(stop_reason)}"
    assert len(stop_reason_cache) == kv_cache.batch_size, f"Expected stop_reason length {kv_cache.batch_size}, but got {len(stop_reason_cache)}"