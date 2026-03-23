import torch, pytest, logging

from llm_engine import PagedKVCache, BlockTable, MemoryAllocator, PagedCacheContext

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

num_blocks = 16
n_layers = 3
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

def test_prefill_single_sequence(paged_cache_context):
    
    seq_ids = ['seq_0']
    seq_lens = [0]
    num_blocks_per_seq = [2]
    T_new = 5
    
    cache = paged_cache_context(seq_ids = seq_ids, 
                              seq_lens = seq_lens, 
                              num_blocks_per_seq = num_blocks_per_seq)
    
    k_new = torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device)
    v_new = torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device)
    
    k_full, v_full = cache.update_cache(layer_idx = 0, k = k_new, v = v_new)
    
    assert k_full.shape == (1, n_heads, T_new, head_dim), f"Expected k_full shape to be {(1, n_heads, T_new, head_dim)}, but got {k_full.shape}"
    assert v_full.shape == (1, n_heads, T_new, head_dim), f"Expected v_full shape to be {(1, n_heads, T_new, head_dim)}, but got {v_full.shape}"
    assert k_full.equal(k_new), "Expected k_full to match k_new"
    assert v_full.equal(v_new), "Expected v_full to match v_new"
    
def test_prefill_then_decode(paged_cache_context):
    
    seq_ids = ['seq_0']
    seq_lens = [0]
    num_blocks_per_seq = [2]
    T_new = 5
    
    cache = paged_cache_context(seq_ids = seq_ids, 
                              seq_lens = seq_lens, 
                              num_blocks_per_seq = num_blocks_per_seq)
    
    k_new_1 = [torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device) for _ in range(n_layers)]
    v_new_1 = [torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device) for _ in range(n_layers)]
    k_new_2 = [torch.randn((1, n_heads, 1    , head_dim), dtype=dtype, device=device) for _ in range(n_layers)]
    v_new_2 = [torch.randn((1, n_heads, 1    , head_dim), dtype=dtype, device=device) for _ in range(n_layers)]
    
    for layer_idx in range(n_layers):
        
        k_full, v_full = cache.update_cache(layer_idx = layer_idx, 
                                            k = k_new_1[layer_idx], 
                                            v = v_new_1[layer_idx])
        
        assert k_full.shape == (1, n_heads, T_new, head_dim), f"Expected k_full shape to be {(1, n_heads, T_new, head_dim)}, but got {k_full.shape}"
        assert v_full.shape == (1, n_heads, T_new, head_dim), f"Expected v_full shape to be {(1, n_heads, T_new, head_dim)}, but got {v_full.shape}"
        assert k_full.equal(k_new_1[layer_idx]), f"Expected k_full to match k_new_1 at layer {layer_idx}"
        assert v_full.equal(v_new_1[layer_idx]), f"Expected v_full to match v_new_2 at layer {layer_idx}"
        
        
    for layer_idx in range(n_layers):
        
        k_full, v_full = cache.update_cache(layer_idx = layer_idx, 
                                            k = k_new_2[layer_idx], 
                                            v = v_new_2[layer_idx])
        
        assert k_full.shape == (1, n_heads, T_new + 1, head_dim), f"Expected k_full shape to be {(1, n_heads, T_new + 1, head_dim)}, but got {k_full.shape}"
        assert v_full.shape == (1, n_heads, T_new + 1, head_dim), f"Expected v_full shape to be {(1, n_heads, T_new + 1, head_dim)}, but got {v_full.shape}"
        assert torch.equal(k_full[:, :, :T_new, :], k_new_1[layer_idx]), f"Expected first {T_new} tokens of k_full to match k_new_1 at layer {layer_idx}"
        assert torch.equal(v_full[:, :, :T_new, :], v_new_1[layer_idx]), f"Expected first {T_new} tokens of v_full to match v_new_1 at layer {layer_idx}"
        assert torch.equal(k_full[:, :, -1:, :], k_new_2[layer_idx]), f"Expected last token of k_full to match k_new_2 at layer {layer_idx}"
        assert torch.equal(v_full[:, :, -1:, :], v_new_2[layer_idx]), f"Expected last token of v_full to match v_new_2 at layer {layer_idx}"
        
def test_seq_len_increments_only_on_last_layer(paged_cache_context):
    
    seq_ids = ['seq_0']
    seq_lens = [0]
    num_blocks_per_seq = [2]
    T_new = 5
    
    cache = paged_cache_context(seq_ids = seq_ids, 
                                seq_lens = seq_lens, 
                                num_blocks_per_seq = num_blocks_per_seq)
    
    k_new = torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device)
    v_new = torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device)
    
    _ = cache.update_cache(layer_idx = 0, k = k_new, v = v_new)
    assert cache.seq_lens[0] == 0, f"Expected seq_lens[0] to remain 0 after updating layer 0, but got {cache.seq_lens[0]}"
    
    k_new = torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device)
    v_new = torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device)
    
    _ = cache.update_cache(layer_idx = 1, k = k_new, v = v_new)
    assert cache.seq_lens[0] == 0, f"Expected seq_lens[0] to remain 0 after updating layer 1, but got {cache.seq_lens[0]}"
    
    k_new = torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device)
    v_new = torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device)
    
    _ = cache.update_cache(layer_idx = 2, k = k_new, v = v_new)
    assert cache.seq_lens[0] == T_new, f"Expected seq_lens[0] to update to {T_new} after updating layer 2, but got {cache.seq_lens[0]}"
    
def test_seq_len_property(paged_cache_context):
    
    seq_ids = ['seq_0', 'seq_1']
    seq_lens = [10, 20]
    num_blocks_per_seq = [2, 3]
    
    cache = paged_cache_context(seq_ids = seq_ids, 
                                seq_lens = seq_lens, 
                                num_blocks_per_seq = num_blocks_per_seq)
    
    assert cache.seq_len == max(seq_lens), f"Expected seq_lens property to return max of seq_lens list, but got {cache.seq_lens}"
    
def test_two_sequences_same_length(paged_cache_context):
    
    seq_ids = ['seq_0', 'seq_1']
    seq_lens = [0, 0]
    num_blocks_per_seq = [1, 1]
    T_new = [3, 3]
    
    cache = paged_cache_context(seq_ids = seq_ids, 
                                seq_lens = seq_lens, 
                                num_blocks_per_seq = num_blocks_per_seq)
    
    k_new_0 = torch.randn((n_heads, T_new[0], head_dim), dtype=dtype, device=device)
    v_new_0 = torch.randn((n_heads, T_new[0], head_dim), dtype=dtype, device=device)
    k_new_1 = torch.randn((n_heads, T_new[1], head_dim), dtype=dtype, device=device)
    v_new_1 = torch.randn((n_heads, T_new[1], head_dim), dtype=dtype, device=device)
    
    k_new = torch.stack([k_new_0, k_new_1], dim=0)
    v_new = torch.stack([v_new_0, v_new_1], dim=0)
    
    k_full, v_full = cache.update_cache(layer_idx = 0, k = k_new, v = v_new)
    
    assert k_full.shape == (2, n_heads, max(T_new), head_dim), f"Expected k_full shape to be {(2, n_heads, max(T_new), head_dim)}, but got {k_full.shape}"
    assert v_full.shape == (2, n_heads, max(T_new), head_dim), f"Expected v_full shape to be {(2, n_heads, max(T_new), head_dim)}, but got {v_full.shape}"
    assert torch.equal(k_full[0, :, :T_new[0], :], k_new_0), "Expected first sequence in k_full to match k_new_0"
    assert torch.equal(v_full[0, :, :T_new[0], :], v_new_0), "Expected first sequence in v_full to match v_new_0"
    assert torch.equal(k_full[1, :, :T_new[1], :], k_new_1), "Expected second sequence in k_full to match k_new_1"
    assert torch.equal(v_full[1, :, :T_new[1], :], v_new_1), "Expected second sequence in v_full to match v_new_1"
    
def test_block_boundary_crossing(paged_cache_context):
    
    seq_ids = ['seq_0']
    seq_lens = [0]
    num_blocks_per_seq = [2]
    T_new = 6
    
    cache = paged_cache_context(seq_ids = seq_ids, 
                                seq_lens = seq_lens, 
                                num_blocks_per_seq = num_blocks_per_seq)
    
    k_new = torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device)
    v_new = torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device)
    
    k_full, v_full = cache.update_cache(layer_idx = 0, k = k_new, v = v_new)
    
    assert cache.block_table.num_blocks(seq_ids[0]) == num_blocks_per_seq[0], f"Expected number of blocks allocated for {seq_ids[0]} to be {num_blocks_per_seq[0]}, but got {cache.block_table.num_blocks(seq_ids[0])}"
    assert k_full.shape == (1, n_heads, T_new, head_dim), f"Expected k_full shape to be {(1, n_heads, T_new, head_dim)}, but got {k_full.shape}"
    assert v_full.shape == (1, n_heads, T_new, head_dim), f"Expected v_full shape to be {(1, n_heads, T_new, head_dim)}, but got {v_full.shape}"
    assert k_full.equal(k_new), "Expected k_full to match k_new even when crossing block boundary"
    assert v_full.equal(v_new), "Expected v_full to match v_new even when crossing block boundary"
    
def test_auto_allocate_on_block_overflow(paged_cache_context):
    
    seq_ids = ['seq_0']
    seq_lens = [0]
    num_blocks_per_seq = [1]
    T_new = 6
    
    cache = paged_cache_context(seq_ids = seq_ids, 
                                seq_lens = seq_lens, 
                                num_blocks_per_seq = num_blocks_per_seq)
    
    k_new = torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device)
    v_new = torch.randn((1, n_heads, T_new, head_dim), dtype=dtype, device=device)
    
    k_full, v_full = cache.update_cache(layer_idx = 0, k = k_new, v = v_new)
    
    assert k_full.shape == (1, n_heads, T_new, head_dim), f"Expected k_full shape to be {(1, n_heads, T_new, head_dim)}, but got {k_full.shape}"
    assert v_full.shape == (1, n_heads, T_new, head_dim), f"Expected v_full shape to be {(1, n_heads, T_new, head_dim)}, but got {v_full.shape}"
    assert k_full.equal(k_new), "Expected k_full to match k_new even when auto-allocating new block"
    assert v_full.equal(v_new), "Expected v_full to match v_new even when auto-allocating new block"
    assert cache.block_table.num_blocks(seq_ids[0]) == num_blocks_per_seq[0] + 1, f"Expected number of blocks allocated for {seq_ids[0]} to auto-increment to 2 after block overflow, but got {cache.block_table.num_blocks(seq_ids[0])}"