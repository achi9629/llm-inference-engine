import torch, pytest
from llm_engine import KVCache

batch_size = 2
n_layers = 3
n_heads = 4
head_dim = 5
max_seq_len = 32
dtype = torch.float32
device = 'cpu'

@pytest.fixture
def kv_cache():
    return KVCache(batch_size = batch_size,
                    n_layers = n_layers,
                    n_heads = n_heads,
                    head_dim = head_dim,
                    max_seq_len = max_seq_len,
                    dtype = dtype,
                    device = device)

def test_initialization_shapes_and_defaults(kv_cache):
    
    if not isinstance(kv_cache.k_cache, list):
        raise AssertionError("k_cache should be a list")
    if not isinstance(kv_cache.v_cache, list):
        raise AssertionError("v_cache should be a list")
    assert len(kv_cache.k_cache) == n_layers, "KVCache should have one entry per layer"
    assert len(kv_cache.v_cache) == n_layers, "KVCache should have one entry per layer"
    
    for layer_k_cache, layer_v_cache in zip(kv_cache.k_cache, kv_cache.v_cache):
        assert layer_k_cache.shape == (batch_size, n_heads, max_seq_len, head_dim), f"Expected k_cache shape {(batch_size, n_heads, max_seq_len, head_dim)}, got {layer_k_cache.shape}"
        assert layer_v_cache.shape == (batch_size, n_heads, max_seq_len, head_dim), f"Expected v_cache shape {(batch_size, n_heads, max_seq_len, head_dim)}, got {layer_v_cache.shape}"
        assert layer_k_cache.dtype == dtype, f"Expected k_cache dtype {dtype}, got {layer_k_cache.dtype}"
        assert layer_v_cache.dtype == dtype, f"Expected v_cache dtype {dtype}, got {layer_v_cache.dtype}"
        assert layer_k_cache.device.type == device, f"Expected k_cache device {device}, got {layer_k_cache.device.type}"
        assert layer_v_cache.device.type == device, f"Expected v_cache device {device}, got {layer_v_cache.device.type}"
        assert kv_cache.seq_len == 0, f"Expected initial seq_len to be 0 initially, got {kv_cache.seq_len}"
        assert torch.all(layer_k_cache == 0), "Expected k_cache to be initialized to zeros"
        assert torch.all(layer_v_cache == 0), "Expected v_cache to be initialized to zeros"
        
def test_single_token_update(kv_cache):
    
    # update layer 0 with T_new=1
    layer_idx = 0
    k_new = torch.randn(batch_size, n_heads, 1, head_dim, dtype=dtype, device=device)
    v_new = torch.randn(batch_size, n_heads, 1, head_dim, dtype=dtype, device=device)
    
    k, v = kv_cache.update_cache(layer_idx, k_new, v_new)
    assert k.shape == (batch_size, n_heads, 1, head_dim), f"Expected returned k shape {(batch_size, n_heads, 1, head_dim)}, got {k.shape}"
    assert v.shape == (batch_size, n_heads, 1, head_dim), f"Expected returned v shape {(batch_size, n_heads, 1, head_dim)}, got {v.shape}"
    assert kv_cache.seq_len == 0, f"Expected seq_len to remain 0 after single token update to layer 0, got {kv_cache.seq_len}"
    
def test_prefill_all_layers(kv_cache):
    
    # update all 3 layers with T_new=10
    T_new = 10
    for layer_idx in range(n_layers):
        k_new = torch.randn(batch_size, n_heads, T_new, head_dim, dtype=dtype, device=device)
        v_new = torch.randn(batch_size, n_heads, T_new, head_dim, dtype=dtype, device=device)
        k, v = kv_cache.update_cache(layer_idx, k_new, v_new)
        
        assert k.shape == (batch_size, n_heads, T_new, head_dim), f"Expected returned k shape {(batch_size, n_heads, T_new, head_dim)}, got {k.shape}"
        assert v.shape == (batch_size, n_heads, T_new, head_dim), f"Expected returned v shape {(batch_size, n_heads, T_new, head_dim)}, got {v.shape}"
        
    assert kv_cache.seq_len == T_new, f"Expected seq_len to be {T_new} after pre-filling all layers, got {kv_cache.seq_len}"
    
def test_prefill_then_decode(kv_cache):
    
    k_new, v_new = [], []
    for layer_idx in range(n_layers):
        k_temp = torch.randn(batch_size, n_heads, 10, head_dim, dtype=dtype, device=device)
        v_temp = torch.randn(batch_size, n_heads, 10, head_dim, dtype=dtype, device=device)
        _, _ = kv_cache.update_cache(layer_idx, k_temp, v_temp)
        k_new.append(k_temp)
        v_new.append(v_temp)
    
    # now decode with T_new = 1 for each layer
    for layer_idx in range(n_layers):
        k_temp = torch.randn(batch_size, n_heads, 1, head_dim, dtype = dtype, device = device)
        v_temp = torch.randn(batch_size, n_heads, 1, head_dim, dtype = dtype, device = device)
        _, _ = kv_cache.update_cache(layer_idx, k_temp, v_temp)
        
    for layer_idx in range(n_layers):
        k_cache_layer = kv_cache.k_cache[layer_idx]
        v_cache_layer = kv_cache.v_cache[layer_idx]

        assert kv_cache.seq_len == 11, f"Expected seq_len to be 11 after pre-filling and decoding, got {kv_cache.seq_len}"
        assert k_cache_layer[:, :, :11, :].shape == (batch_size, n_heads, 11, head_dim), f"Expected k_cache layer shape {(batch_size, n_heads, 11, head_dim)}, got {k_cache_layer.shape}"
        assert v_cache_layer[:, :, :11, :].shape == (batch_size, n_heads, 11, head_dim), f"Expected v_cache layer shape {(batch_size, n_heads, 11, head_dim)}, got {v_cache_layer.shape}"
        
        assert k_cache_layer[:, :, :10, :].equal(k_new[layer_idx]), f"Expected first 10 tokens of k_cache layer {layer_idx} to match pre-filled values"
        assert v_cache_layer[:, :, :10, :].equal(v_new[layer_idx]), f"Expected first 10 tokens of v_cache layer {layer_idx} to match pre-filled values"

def test_values_written_correctly(kv_cache):
    
    start = 0
    end = 5
    for layer_idx in range(n_layers):
        k_temp = torch.ones(batch_size, n_heads, end - start, head_dim, dtype = dtype, device = device)
        v_temp = torch.ones(batch_size, n_heads, end - start, head_dim, dtype = dtype, device = device)
        _, _ = kv_cache.update_cache(layer_idx, k_temp, v_temp)
        
    for layer_idx in range(n_layers):
        
        assert torch.all(kv_cache.k_cache[layer_idx][:, :, start: end, :] == 1), f"Expected k_cache values for layer {layer_idx} to be 1 in the updated range"
        assert torch.all(kv_cache.v_cache[layer_idx][:, :, start: end, :] == 1), f"Expected v_cache values for layer {layer_idx} to be 1 in the updated range"
        
        assert torch.all(kv_cache.k_cache[layer_idx][:, :, end:, :] == 0), f"Expected k_cache values for layer {layer_idx} to be 0 outside the updated range"
        assert torch.all(kv_cache.v_cache[layer_idx][:, :, end:, :] == 0), f"Expected v_cache values for layer {layer_idx} to be 0 outside the updated range"

def test_seq_len_increments_only_on_last_layer(kv_cache):
    
    for layer_idx in range(n_layers - 1):
        
        k_new = torch.randn(batch_size, n_heads, 5, head_dim, dtype=dtype, device=device)
        v_new = torch.randn(batch_size, n_heads, 5, head_dim, dtype=dtype, device=device)
        
        _, _ = kv_cache.update_cache(layer_idx, k_new, v_new)
        
    assert kv_cache.seq_len == 0, f"Expected seq_len to remain 0 after updates to all but last layer, got {kv_cache.seq_len}"
    
    k_new = torch.randn(batch_size, n_heads, 5, head_dim, dtype=dtype, device=device)
    v_new = torch.randn(batch_size, n_heads, 5, head_dim, dtype=dtype, device=device)
    _, _ = kv_cache.update_cache(n_layers - 1, k_new, v_new)
    
    assert kv_cache.seq_len == 5, f"Expected seq_len to update to 5 after update to last layer, got {kv_cache.seq_len}"
    
def test_reset(kv_cache):
    
    for layer_idx in range(n_layers):
        k_new = torch.randn(batch_size, n_heads, 5, head_dim, dtype=dtype, device=device)
        v_new = torch.randn(batch_size, n_heads, 5, head_dim, dtype=dtype, device=device)
        _, _ = kv_cache.update_cache(layer_idx, k_new, v_new)
        
    kv_cache.reset_cache()
    
    assert kv_cache.seq_len == 0, f"Expected seq_len to reset to 0 after reset_cache, got {kv_cache.seq_len}"
    for layer_k_cache, layer_v_cache in zip(kv_cache.k_cache, kv_cache.v_cache):
        
        assert torch.all(layer_k_cache == 0), "Expected k_cache to be reset to zeros after reset_cache"
        assert torch.all(layer_v_cache == 0), "Expected v_cache to be reset to zeros after reset_cache"
    
        
    