import torch, pytest

from llm_engine import MultiHeadAttention, KVCache

# constants
batch_size = 2
n_layers = 5
n_embd = 64
n_heads = 4
n_ctx = 32
head_dim = n_embd // n_heads
atol = 1e-7

@pytest.fixture
def attention():
    
    atten_layers = []
    for _ in range(n_layers):
        attn = MultiHeadAttention(n_embd = n_embd, 
                                n_heads = n_heads,
                                n_ctx = n_ctx,
                )
        attn.eval()
        atten_layers.append(attn)
    return atten_layers
    
@pytest.fixture
def kv_cache():
    return KVCache(batch_size = batch_size,
                   n_layers = n_layers,
                   n_heads = n_heads,
                   head_dim = head_dim,
                   max_seq_len = n_ctx,
                   dtype = torch.float32,
                   device = 'cpu',
            )
    
def test_no_cache_output_shape(attention):
    
    x_inp = torch.randn(batch_size, 10, n_embd)
    
    x = x_inp.clone()
    for layer_idx in range(n_layers):
        x = attention[layer_idx](x)
    out = x.clone()
    
    assert out.shape == (batch_size, 10, n_embd), f"Expected output shape {(batch_size, 10, n_embd)}, got {out.shape}"
    
def test_no_cache_vs_cache_prefill_identical(attention, kv_cache):
    
    x_inp = torch.randn(batch_size, 10, n_embd)
    
    x = x_inp.clone()
    for layer_idx in range(n_layers):
        x = attention[layer_idx](x)
    out_with_no_cache = x.clone()
    
    x = x_inp.clone()
    for layer_idx in range(n_layers):
        x = attention[layer_idx](x, layer_idx = layer_idx, kv_cache = kv_cache)
    out_with_cache = x.clone()
    
    assert torch.allclose(out_with_no_cache, out_with_cache, atol = atol), "Outputs with and without cache prefill should be close"
    
def test_prefill_then_decode_matches_full_pass(attention, kv_cache):
    
    x_full = torch.randn(batch_size, 11, n_embd)
    
    x = x_full.clone()
    for layer_idx in range(n_layers):
        x = attention[layer_idx](x)
    out_full = x.clone()
    last_pos_full = out_full[:, -1, :].clone()
    
    x = x_full[:, :10, :].clone()
    for layer_idx in range(n_layers):
        x = attention[layer_idx](x, layer_idx = layer_idx, kv_cache = kv_cache)
    
    x = x_full[:, 10:, :].clone()
    for layer_idx in range(n_layers):
        x = attention[layer_idx](x, layer_idx = layer_idx, kv_cache = kv_cache)
    last_pos_cache = x.clone()
    
    assert torch.allclose(last_pos_cache, last_pos_full, atol = atol), "Output from decode step should match last position of full pass"
     
    
def test_cache_state_after_updates(attention, kv_cache):
    
    x_inp = torch.randn(batch_size, 10, n_embd)
    for layer_idx in range(n_layers):
        _ = attention[layer_idx](x_inp, layer_idx = layer_idx, kv_cache = kv_cache)
    assert kv_cache.seq_len == 10, f"Expected cache seq_len to be 10 after prefill, got {kv_cache.seq_len}"
    
    x_inp = torch.randn(batch_size, 1, n_embd)
    for layer_idx in range(n_layers):
        _ = attention[layer_idx](x_inp, layer_idx = layer_idx, kv_cache = kv_cache)
    assert kv_cache.seq_len == 11, f"Expected cache seq_len to be 11 after decode step, got {kv_cache.seq_len}"
    

    
    
    
    