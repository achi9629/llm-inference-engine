import torch, pytest # type: ignore

from llm_engine import PagedKVCache

num_blocks = 8
n_layers = 2
n_heads = 4
block_size = 4
head_dim = 8
dtype = torch.float32
device = 'cpu'

@pytest.fixture
def paged_kv_cache():
    
    return PagedKVCache(num_blocks = num_blocks, 
                        n_layers = n_layers,
                        n_heads = n_heads,
                        block_size =block_size,
                        head_dim = head_dim,
                        dtype = dtype,
                        device = device
            )
    
def test_initialization(paged_kv_cache):
    
    assert len(paged_kv_cache.k_cache) == n_layers, "K cache should have an entry for each layer"
    assert len(paged_kv_cache.v_cache) == n_layers, "V cache should have an entry for each layer"
    
    for layer_idx in range(n_layers):
        assert paged_kv_cache.k_cache[layer_idx].shape == (num_blocks, n_heads, block_size, head_dim), f"K cache for layer {layer_idx} has incorrect shape"
        assert paged_kv_cache.v_cache[layer_idx].shape == (num_blocks, n_heads, block_size, head_dim), f"V cache for layer {layer_idx} has incorrect shape"
        assert torch.all(paged_kv_cache.k_cache[layer_idx] == 0), f"K cache for layer {layer_idx} should be initialized to zeros"
        assert torch.all(paged_kv_cache.v_cache[layer_idx] == 0), f"V cache for layer {layer_idx} should be initialized to zeros"
        
def test_write_and_read(paged_kv_cache):
    
    k_new = torch.randn((n_heads, head_dim), dtype=dtype, device=device)
    v_new = torch.randn((n_heads, head_dim), dtype=dtype, device=device)
    layer_idx = 0
    block_id = 0
    offset = 0
    
    paged_kv_cache.write(layer_idx, block_id, offset, k_new, v_new)
    
    k_out, v_out = paged_kv_cache.read(layer_idx, [block_id])
    
    assert k_out.shape == (1, n_heads, block_size, head_dim), f"Read K has incorrect shape, got {k_out.shape} but expected {(1, n_heads, block_size, head_dim)}"
    assert v_out.shape == (1, n_heads, block_size, head_dim), f"Read V has incorrect shape, got {v_out.shape} but expected {(1, n_heads, block_size, head_dim)}"
    
    assert k_out[0, :, offset, :].equal(k_new), "Written K data should match at the correct offset"
    assert v_out[0, :, offset, :].equal(v_new), "Written V data should match at the correct offset"
    
def test_read_multiple_blocks(paged_kv_cache):
    
    k_new = torch.randn((n_heads, head_dim), dtype=dtype, device=device)
    v_new = torch.randn((n_heads, head_dim), dtype=dtype, device=device)
    layer_idx = 0
    
    block_id, offset = 0, 0
    paged_kv_cache.write(layer_idx, block_id, offset, k_new, v_new)
    
    block_id, offset = 2, 1
    paged_kv_cache.write(layer_idx, block_id, offset, k_new, v_new)
    
    k_out, v_out = paged_kv_cache.read(layer_idx, [0, 2])
    
    assert k_out.shape == (1, n_heads, block_size * 2, head_dim), f"Read K has incorrect shape, got {k_out.shape} but expected {(1, n_heads, block_size * 2, head_dim)}"
    assert v_out.shape == (1, n_heads, block_size * 2, head_dim), f"Read V has incorrect shape, got {v_out.shape} but expected {(1, n_heads, block_size * 2, head_dim)}"
    
    assert k_out[0, :, 0, :].equal(k_new), "First block of read K should match first written K"
    assert v_out[0, :, 0, :].equal(v_new), "First block of read V should match first written V"
    assert k_out[0, :, block_size + 1, :].equal(k_new), "Second block of read K should match second written K"
    assert v_out[0, :, block_size + 1, :].equal(v_new), "Second block of read V should match second written V"
    
def test_reset_blocks(paged_kv_cache):
    
    k_new = torch.randn((n_heads, head_dim), dtype=dtype, device=device)
    v_new = torch.randn((n_heads, head_dim), dtype=dtype, device=device)
    layer_idx = 0
    
    block_id, offset = 0, 0
    paged_kv_cache.write(layer_idx, block_id, offset, k_new, v_new)
    
    block_id, offset = 1, 0
    paged_kv_cache.write(layer_idx, block_id, offset, k_new, v_new)
    
    paged_kv_cache.reset_blocks([0])
    
    k_out, v_out = paged_kv_cache.read(layer_idx, [0])
    
    assert torch.all(k_out == 0), "K cache for reset block should be zeros"
    assert torch.all(v_out == 0), "V cache for reset block should be zeros"
    
    k_out, v_out = paged_kv_cache.read(layer_idx, [1])
    
    assert not torch.all(k_out == 0), "K cache for non-reset block should not be zeros"
    assert not torch.all(v_out == 0), "V cache for non-reset block should not be zeros"