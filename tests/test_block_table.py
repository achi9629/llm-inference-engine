import torch, pytest

from llm_engine import MemoryAllocator, BlockTable

num_blocks = 8
block_size = 4

@pytest.fixture
def block_table():
    allocator = MemoryAllocator(num_blocks = num_blocks)
    return BlockTable(allocator = allocator, block_size = block_size)
    
def test_add_sequence(block_table):
    
    seq_id = 'A'
    block_table.add_sequence(seq_id)
    
    assert block_table.get_block_ids(seq_id) == [], f"Expected no blocks allocated for sequence {seq_id} yet."
    assert block_table.num_blocks(seq_id) == 0, f"Expected zero blocks allocated for sequence {seq_id}."
    

def test_add_duplicate_sequence(block_table):
    
    seq_id = 'A'
    block_table.add_sequence(seq_id)
    with pytest.raises(ValueError, match = f"Sequence ID {seq_id} already exists in the block table."):
        block_table.add_sequence(seq_id)
        
def test_allocate_blocks(block_table):
    
    seq_id = 'A'
    block_nos = 3
    
    block_table.add_sequence(seq_id)
    allocated_blocks = block_table.allocate_blocks(seq_id, block_nos)
    
    assert block_table.get_block_ids(seq_id) == allocated_blocks, f"Expected allocated blocks {allocated_blocks} for sequence {seq_id}."
    assert block_table.num_blocks(seq_id) == block_nos, f"Expected {block_nos} blocks allocated for sequence {seq_id}."
    assert block_table.allocator.num_free_blocks == num_blocks - block_nos, f"Expected {num_blocks - block_nos} free blocks remaining in the allocator."
    
def test_get_physical_block(block_table):
    
    seq_id = 'A'
    block_nos = 3
    
    block_table.add_sequence(seq_id)
    allocated_blocks = block_table.allocate_blocks(seq_id, block_nos)
    
    assert block_table.get_physical_block(seq_id, 0) == allocated_blocks[0], f"Expected physical block {allocated_blocks[0]} for logical block 0 of sequence {seq_id}."
    assert block_table.get_physical_block(seq_id, 1) == allocated_blocks[1], f"Expected physical block {allocated_blocks[1]} for logical block 1 of sequence {seq_id}."
    assert block_table.get_physical_block(seq_id, 2) == allocated_blocks[2], f"Expected physical block {allocated_blocks[2]} for logical block 2 of sequence {seq_id}."
    
def test_get_physical_block_out_of_bounds(block_table):
    
    seq_id = 'A'
    block_nos = 2
    logical_index = 5
    
    block_table.add_sequence(seq_id)
    _ = block_table.allocate_blocks(seq_id, block_nos)
    
    with pytest.raises(ValueError, match = f"Logical index {logical_index} is out of bounds for sequence ID {seq_id}."):
        block_table.get_physical_block(seq_id, logical_index)
        
def test_free_sequence(block_table):
    
    seq_id = 'A'
    block_nos = 3
    
    block_table.add_sequence(seq_id)
    _ = block_table.allocate_blocks(seq_id, block_nos)
    
    
    block_table.free_sequence(seq_id)
    
    assert block_table.allocator.num_free_blocks == num_blocks, f"Expected all blocks to be free after freeing sequence {seq_id}."
    
    with pytest.raises(ValueError, match = f"Sequence ID {seq_id} does not exist in the block table."):
        block_table.get_block_ids(seq_id)
        
def test_unknown_sequence(block_table):
    
    seq_id = 'Z'
    
    with pytest.raises(ValueError, match = f"Sequence ID {seq_id} does not exist in the block table."):
        block_table.get_block_ids(seq_id)