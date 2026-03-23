import torch, pytest
from llm_engine import ContinuousKVCache

batch_size = 2
n_layers = 3
n_heads = 4
head_dim = 5
max_seq_len = 32
dtype = torch.float32
device = 'cpu'

@pytest.fixture
def kv_cache():
    return ContinuousKVCache(batch_size = batch_size,
                             n_layers = n_layers,
                             n_heads = n_heads,
                             head_dim = head_dim,
                             max_seq_len = max_seq_len,
                             dtype = dtype,
                             device = device
                )
    
def test_initialization(kv_cache):
    
    assert kv_cache.seq_len == [0, 0], "Initial sequence lengths should be zero"
    assert len(kv_cache.k_cache) == n_layers, "Number of layers in k_cache should match n_layers"
    assert len(kv_cache.v_cache) == n_layers, "Number of layers in v_cache should match n_layers"
    for layer_idx in range(n_layers):
        
        assert kv_cache.k_cache[layer_idx].shape == (batch_size, n_heads, max_seq_len, head_dim), \
            f"Shape of k_cache for layer {layer_idx} should be {(batch_size, n_heads, max_seq_len, head_dim)}"
        assert kv_cache.v_cache[layer_idx].shape == (batch_size, n_heads, max_seq_len, head_dim), \
            f"Shape of v_cache for layer {layer_idx} should be {(batch_size, n_heads, max_seq_len, head_dim)}"
        assert torch.all(kv_cache.k_cache[layer_idx] == 0), f"k_cache for layer {layer_idx} should be initialized to zeros"
        assert torch.all(kv_cache.v_cache[layer_idx] == 0), f"v_cache for layer {layer_idx} should be initialized to zeros"
        
def test_per_sequence_seq_len(kv_cache):
    
    seq_len = [5, 5]
    T_new = 5
    
    k_new = torch.randn(batch_size, n_heads, T_new, head_dim, dtype=dtype, device=device)
    v_new = torch.randn(batch_size, n_heads, T_new, head_dim, dtype=dtype, device=device)
    
    for layer_idx in range(n_layers):
        kv_cache.update_cache(layer_idx, k_new, v_new)
        if layer_idx < n_layers - 1:
            assert kv_cache.seq_len == [0, 0], f"Sequence lengths should be updated to {seq_len} tille last layer but got {kv_cache.seq_len}"
        else:
            assert kv_cache.seq_len == seq_len, f"Sequence lengths should be updated to {seq_len} after final update but got {kv_cache.seq_len}"
    
        
def test_reset_slot(kv_cache):
    
    T_new = 5
    seq_len = [T_new, T_new]
    k_new = torch.randn(batch_size, n_heads, T_new, head_dim, dtype=dtype, device=device)
    v_new = torch.randn(batch_size, n_heads, T_new, head_dim, dtype=dtype, device=device)
    
    for layer_idx in range(n_layers):
        kv_cache.update_cache(layer_idx, k_new, v_new)

    assert kv_cache.seq_len == seq_len, f"Sequence lengths should be updated to {seq_len} after updates but got {kv_cache.seq_len}"
    
    kv_cache.reset_slot(batch_idx = 0)
    assert kv_cache.seq_len == [0, 5], f"After resetting slot 0, sequence lengths should be [0, 5] but got {kv_cache.seq_len}"
    
    for layer_idx in range(n_layers):
        assert torch.all(kv_cache.k_cache[layer_idx][0] == 0), f"After resetting slot 0, k_cache for slot 0 in layer {layer_idx} should be zeros"
        assert torch.all(kv_cache.v_cache[layer_idx][0] == 0), f"After resetting slot 0, v_cache for slot 0 in layer {layer_idx} should be zeros"
        
        assert torch.all(kv_cache.k_cache[layer_idx][1, :, :seq_len[1], :] != 0), f"After resetting slot 0, k_cache for slot 1 in layer {layer_idx} should not be zeros"
        assert torch.all(kv_cache.v_cache[layer_idx][1, :, :seq_len[1], :] != 0), f"After resetting slot 0, v_cache for slot 1 in layer {layer_idx} should not be zeros"
        
def test_reset_cache(kv_cache):
    
    T_new = 5
    seq_len = [T_new, T_new]
    k_new = torch.randn(batch_size, n_heads, T_new, head_dim, dtype=dtype, device=device)
    v_new = torch.randn(batch_size, n_heads, T_new, head_dim, dtype=dtype, device=device)
    
    for layer_idx in range(n_layers):
        kv_cache.update_cache(layer_idx, k_new, v_new)
        
    assert kv_cache.seq_len == seq_len, f"Sequence lengths should be updated to {seq_len} after updates but got {kv_cache.seq_len}"
    
    kv_cache.reset_cache()
    assert kv_cache.seq_len == [0, 0], f"After resetting cache, sequence lengths should be [0, 0] but got {kv_cache.seq_len}"
    for layer_idx in range(n_layers):
        assert torch.all(kv_cache.k_cache[layer_idx] == 0), f"After resetting cache, k_cache for layer {layer_idx} should be zeros"
        assert torch.all(kv_cache.v_cache[layer_idx] == 0), f"After resetting cache, v_cache for layer {layer_idx} should be zeros"
        
def test_update_cache_per_batch_idx(kv_cache):
    
    
    T_new = 3
    seq_len = [T_new, T_new]
    
    k_new_1 = torch.randn(batch_size, n_heads, T_new, head_dim, dtype=dtype, device=device)
    v_new_1 = torch.randn(batch_size, n_heads, T_new, head_dim, dtype=dtype, device=device)
    
    for layer_idx in range(n_layers):
        kv_cache.update_cache(layer_idx, k_new_1, v_new_1)
        
    T_new = 1
    seq_len = [T_new, T_new]
    
    k_new_2 = torch.randn(batch_size, n_heads, T_new, head_dim, dtype=dtype, device=device)
    v_new_2 = torch.randn(batch_size, n_heads, T_new, head_dim, dtype=dtype, device=device)
    
    for layer_idx in range(n_layers):
        k_cache, v_cache = kv_cache.update_cache(layer_idx, k_new_2, v_new_2)
        
        
    assert kv_cache.seq_len == [4, 4], f"After second update, sequence lengths should be {seq_len} but got {kv_cache.seq_len}"
    assert k_cache.shape == (batch_size, n_heads, 4, head_dim), f"Shape of returned k_cache should be {(batch_size, n_heads, 4, head_dim)} but got {k_cache.shape}"
    assert v_cache.shape == (batch_size, n_heads, 4, head_dim), f"Shape of returned v_cache should be {(batch_size, n_heads, 4, head_dim)} but got {v_cache.shape}"
    
    assert k_cache[:, :, :3, :].equal(k_new_1), "The first 3 tokens in k_cache should match k_new_1"
    assert v_cache[:, :, :3, :].equal(v_new_1), "The first 3 tokens in v_cache should match v_new_1"
    assert k_cache[:, :, 3:, :].equal(k_new_2), "The 4th token in k_cache should match k_new_2"
    assert v_cache[:, :, 3:, :].equal(v_new_2), "The 4th token in v_cache should match v_new_2"