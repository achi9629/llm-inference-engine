import torch, pytest

from llm_engine import MultiHeadAttention, KVCache, \
                       PagedCacheContext, PagedKVCache, MemoryAllocator, \
                       BlockTable, MistralAttention

# constants
num_blocks = 32
block_size = 16
batch_size = 32
n_layers = 5
n_embd = 512
n_heads = 32
n_kv_heads = 8
n_ctx = 32
head_dim = n_embd // n_heads
atol = 1e-7
dtype = torch.float32

@pytest.fixture
def attention():
    
    atten_layers = []
    for _ in range(n_layers):
        attn = MultiHeadAttention(n_embd = n_embd, 
                                n_heads = n_heads,
                                n_ctx = n_ctx,
                )
        attn.eval()
        attn.to(dtype)
        atten_layers.append(attn)
    return atten_layers
    
@pytest.fixture
def kv_cache():
    return KVCache(batch_size = batch_size,
                   n_layers = n_layers,
                   n_heads = n_heads,
                   head_dim = head_dim,
                   max_seq_len = n_ctx,
                   dtype = dtype,
                   device = 'cpu',
            )
    
def test_no_cache_output_shape(attention):
    
    x_inp = torch.randn(batch_size, 10, n_embd, dtype = dtype)
    
    x = x_inp.clone()
    for layer_idx in range(n_layers):
        x = attention[layer_idx](x)
    out = x.clone()
    
    assert out.shape == (batch_size, 10, n_embd), f"Expected output shape {(batch_size, 10, n_embd)}, got {out.shape}"
    
def test_no_cache_vs_cache_prefill_identical(attention, kv_cache):
    
    x_inp = torch.randn(batch_size, 10, n_embd, dtype = dtype)
    
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
    
    x_full = torch.randn(batch_size, 11, n_embd, dtype = dtype)
    
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
    last_pos_cache = x[:, -1, :].clone()

    assert torch.allclose(last_pos_cache, last_pos_full, atol = atol), "Output from decode step should match last position of full pass"

def test_cache_state_after_updates(attention, kv_cache):
    
    x_inp = torch.randn(batch_size, 10, n_embd, dtype = dtype)
    for layer_idx in range(n_layers):
        _ = attention[layer_idx](x_inp, layer_idx = layer_idx, kv_cache = kv_cache)
    assert kv_cache.seq_len == 10, f"Expected cache seq_len to be 10 after prefill, got {kv_cache.seq_len}"
    
    x_inp = torch.randn(batch_size, 1, n_embd, dtype = dtype)
    for layer_idx in range(n_layers):
        _ = attention[layer_idx](x_inp, layer_idx = layer_idx, kv_cache = kv_cache)
    assert kv_cache.seq_len == 11, f"Expected cache seq_len to be 11 after decode step, got {kv_cache.seq_len}"
    
# Mistral Attention tests
@pytest.fixture
def mistral_attention():
    
    atten_layers = []
    for _ in range(n_layers):
        attn = MistralAttention(n_embd = n_embd, 
                                n_heads = n_heads,
                                n_kv_heads = n_kv_heads,
                                n_ctx = n_ctx,
                )
        attn.eval()
        atten_layers.append(attn)
    return atten_layers

@pytest.fixture
def kv_group_cache():
    return KVCache(batch_size = batch_size,
                   n_layers = n_layers,
                   n_heads = n_kv_heads,
                   head_dim = head_dim,
                   max_seq_len = n_ctx,
                   dtype = dtype,
                   device = 'cpu',
            )
    
@pytest.fixture
def paged_kv_cache():
    
    return PagedKVCache(num_blocks = num_blocks, 
                        n_layers = n_layers,
                        n_heads = n_kv_heads,
                        block_size = block_size,
                        head_dim = head_dim,
                        dtype = dtype,
                        device = 'cpu'
            ) 
    
@pytest.fixture
def block_table():
    
    allocator = MemoryAllocator(num_blocks = num_blocks)
    return BlockTable(allocator = allocator, 
                      block_size = block_size)

@pytest.fixture
def paged_cache_context(paged_kv_cache, block_table):
    """Returns a factory function that creates a PagedCacheContext with pre-allocated blocks."""
    
    def _make(seq_ids, seq_lens, num_blocks_per_seq):
        
        for seq_id, n in zip(seq_ids, num_blocks_per_seq):
            
            block_table.add_sequence(seq_id)
            block_table.allocate_blocks(seq_id, n)
            
        return PagedCacheContext(paged_kv_cache = paged_kv_cache,
                                 block_table = block_table,
                                 seq_ids = seq_ids,
                                 seq_lens = seq_lens,
                                 block_size = block_size)
        
    return _make

def test_mistral_no_cache_output_shape(mistral_attention):
    
    x_inp = torch.randn(batch_size, 10, n_embd, dtype = dtype)
    
    x = x_inp.clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x)
    out = x.clone()
    
    assert out.shape == (batch_size, 10, n_embd), f"Expected output shape {(batch_size, 10, n_embd)}, got {out.shape}"
    
def test_mistral_no_cache_vs_cache_prefill_identical(mistral_attention, kv_group_cache):
    
    x_inp = torch.randn(batch_size, 10, n_embd, dtype = dtype)
    
    x = x_inp.clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x)
    out_with_no_cache = x.clone()
    
    x = x_inp.clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = kv_group_cache)
    out_with_cache = x.clone()
    
    assert torch.allclose(out_with_no_cache, out_with_cache, atol = atol), "Outputs with and without cache prefill should be close"
    
def test_mistral_prefill_then_decode_matches_full_pass(mistral_attention, kv_group_cache):
    
    x_full = torch.randn(batch_size, 11, n_embd, dtype = dtype)
    
    x = x_full.clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x)
    out_full = x.clone()
    last_pos_full = out_full[:, -1, :].clone()
    
    x = x_full[:, :10, :].clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = kv_group_cache)
    
    x = x_full[:, 10:, :].clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = kv_group_cache)
    last_pos_cache = x[:, -1, : ].clone()
    
    assert torch.allclose(last_pos_cache, last_pos_full, atol = atol), "Output from decode step should match last position of full pass"
    
def test_mistral_kv_cache_vs_paged_kv_cache(mistral_attention, 
                                            paged_cache_context, 
                                            kv_group_cache):
    
    # x_inp shape: (batch_size, seq_len, n_embd)
    x_inp = torch.randn(batch_size, 10, n_embd, dtype = dtype)
    
    x = x_inp.clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = kv_group_cache)
    out_with_std_cache = x.clone()
    
    seq_ids = [f"seq_{i}" for i in range(batch_size)]
    seq_lens = [0] * batch_size
    num_blocks_per_seq = [1] * batch_size
    
    cache = paged_cache_context(seq_ids = seq_ids, 
                              seq_lens = seq_lens, 
                              num_blocks_per_seq = num_blocks_per_seq)
    
    x = x_inp.clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = cache)
    out_with_paged_cache = x.clone()
    
    assert torch.allclose(out_with_std_cache, out_with_paged_cache, atol = atol), "Outputs with standard KV cache and paged KV cache should be close"
    
def test_mistral_cache_length_growth(mistral_attention, 
                                     kv_group_cache,
                                     paged_cache_context):
    
    x_inp = torch.randn(batch_size, 10, n_embd, dtype = dtype)
    
    x = x_inp[:, : 5, :].clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = kv_group_cache)
    assert kv_group_cache.seq_len == 5, f"Expected cache seq_len to be 5 after prefill, got {kv_group_cache.seq_len}"
    
    for i in range(5, 10):
        x = x_inp[:, i: i+1, :].clone()
        for layer_idx in range(n_layers):
            x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = kv_group_cache)
        assert kv_group_cache.seq_len == i + 1, f"Expected cache seq_len to be {i + 1} after decode step, got {kv_group_cache.seq_len}"
    
    seq_ids = [f"seq_{i}" for i in range(batch_size)]
    seq_lens = [0] * batch_size
    num_blocks_per_seq = [1] * batch_size
    
    cache = paged_cache_context(seq_ids = seq_ids, 
                              seq_lens = seq_lens, 
                              num_blocks_per_seq = num_blocks_per_seq)
    
    x = x_inp[:, : 5, :].clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = cache)
    assert cache.seq_len == 5, f"Expected paged cache seq_len to be 5 after prefill, got {cache.seq_len}"
    
    for i in range(5, 10):
        x = x_inp[:, i: i+1, :].clone()
        for layer_idx in range(n_layers):
            x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = cache)
        assert cache.seq_len == i + 1, f"Expected paged cache seq_len to be {i + 1} after decode step, got {cache.seq_len}"
    
def test_mistral_chunked_prefill_parity(mistral_attention,
                                        kv_group_cache, 
                                        paged_cache_context):
    
    x_inp = torch.randn(batch_size, 10, n_embd, dtype = dtype)
    
    # Standard KV Cache test
    # One-shop prefill
    x = x_inp.clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = kv_group_cache)
    out_one_shot = x.clone()
    
    # Reset cache state for chunked prefill test
    kv_group_cache.reset_cache()
    
    # Chunked prefill (first 4 tokens, then remaining 6 tokens)
    x = x_inp[:, :4, :].clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = kv_group_cache)
        
    x = x_inp[:, 4:, :].clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = kv_group_cache)
    out_chunked = x.clone()

    assert torch.allclose(out_one_shot[:, 4:, :], out_chunked, atol = atol), "Outputs from one-shot prefill and chunked prefill should be close for standard KV cache"
    
    # Paged KV Cache test
    # One-shot prefill
    seq_ids = [f"seq_{i}" for i in range(batch_size)]
    seq_lens = [0] * batch_size
    num_blocks_per_seq = [1] * batch_size
    
    cache = paged_cache_context(seq_ids = seq_ids, 
                                seq_lens = seq_lens, 
                                num_blocks_per_seq = num_blocks_per_seq)
    
    x = x_inp.clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = cache)
    out_one_shot = x.clone()
    
    # Reset cache state for chunked prefill test
    cache.reset_blocks() # Reset the PagedCacheContext's block tracking after freeing blocks in the block table.
            
    # Clean up allocated blocks for the current generation after generation is complete.
    for seq_id in seq_ids:
        cache.block_table.free_sequence(seq_id = seq_id)
        
    # Re-create cache context to ensure a clean state for the chunked prefill test.
    cache = paged_cache_context(seq_ids = seq_ids, 
                                seq_lens = seq_lens, 
                                num_blocks_per_seq = num_blocks_per_seq)

    # Chunked prefill (first 4 tokens, then remaining 6 tokens)
    x = x_inp[:, :4, :].clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = cache)
        
    x = x_inp[:, 4:, :].clone()
    for layer_idx in range(n_layers):
        x = mistral_attention[layer_idx](x, layer_idx = layer_idx, kv_cache = cache)
    out_chunked = x.clone()
    
    assert torch.allclose(out_one_shot[:, 4:, :], out_chunked, atol = atol), "Outputs from one-shot prefill and chunked prefill should be close for paged KV cache"