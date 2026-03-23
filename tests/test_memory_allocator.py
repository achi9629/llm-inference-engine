import pytest, logging
from collections import deque

from llm_engine import MemoryAllocator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

num_blocks = 8

@pytest.fixture
def memory_allocator():
    return MemoryAllocator(num_blocks = num_blocks)

def test_initialization(memory_allocator):
    
    assert memory_allocator.num_free_blocks == num_blocks, \
        f"Expected {num_blocks} free blocks, but got {memory_allocator.num_free_blocks}"
    assert memory_allocator.free_blocks == deque(range(num_blocks)), \
        f"Expected free blocks to be {deque(range(num_blocks))}, but got {memory_allocator.free_blocks}"
    assert memory_allocator.allocated_blocks == set(), \
        f"Expected allocated blocks to be an empty set, but got {memory_allocator.allocated_blocks}"
       
def test_allocate(memory_allocator):
    
    blocks_to_allocate = 3
    
    allocated_blocks = memory_allocator.allocate(blocks_to_allocate)
    
    assert memory_allocator.num_free_blocks == num_blocks - len(allocated_blocks), \
        f"Expected {num_blocks - len(allocated_blocks)} free blocks, but got {memory_allocator.num_free_blocks}"
    for block in allocated_blocks:
        assert block in memory_allocator.allocated_blocks, \
            f"Block {block} should be in allocated blocks, but it's not."
    
def test_allocate_all(memory_allocator):
    
    blocks_to_allocate = 8
    
    allocated_blocks = memory_allocator.allocate(blocks_to_allocate)
    
    assert memory_allocator.num_free_blocks == 0, \
        f"Expected 0 free blocks, but got {memory_allocator.num_free_blocks}"
    for block in allocated_blocks:
        assert block in memory_allocator.allocated_blocks, \
            f"Block {block} should be in allocated blocks, but it's not."
            
def test_allocate_insufficient(memory_allocator):
    
    blocks_to_allocate = 9
    
    with pytest.raises(MemoryError, match = "Not enough free blocks available"):
        memory_allocator.allocate(blocks_to_allocate)
        
def test_free(memory_allocator):
    
    blocks_to_allocate = 3
    
    allocated_blocks = memory_allocator.allocate(blocks_to_allocate)
    memory_allocator.free(allocated_blocks)
    
    assert memory_allocator.num_free_blocks == num_blocks, \
        f"Expected {num_blocks} free blocks after freeing, but got {memory_allocator.num_free_blocks}"
    for block in allocated_blocks:
        assert block not in memory_allocator.allocated_blocks, \
            f"Block {block} should not be in allocated blocks after freeing, but it is."
            
def test_reuse_after_free(memory_allocator):
    
    blocks_to_allocate_1 = 3
    allocated_blocks_1 = memory_allocator.allocate(blocks_to_allocate_1)
    memory_allocator.free(allocated_blocks_1[ : 2])
    
    assert memory_allocator.free_blocks == deque([3, 4, 5, 6, 7, 0, 1]), \
        f"Expected free blocks to be [3, 4, 5, 6, 7, 0, 1], but got {memory_allocator.free_blocks}"
    
    blocks_to_allocate_2 = 2
    allocated_blocks_2 = memory_allocator.allocate(blocks_to_allocate_2)
    
    assert allocated_blocks_2 == [3, 4], \
        f"Expected allocated blocks to be [3, 4], but got {allocated_blocks_2}"
    assert memory_allocator.num_free_blocks == num_blocks - blocks_to_allocate_1, \
        f"Expected {num_blocks - blocks_to_allocate_1} free blocks, but got {memory_allocator.num_free_blocks}"
        
def test_free_invalid_block_id(memory_allocator):
    
    with pytest.raises(ValueError, match="Invalid block ID: -1"):
        memory_allocator.free([-1])
        
    with pytest.raises(ValueError, match="Invalid block ID: 8"):
        memory_allocator.free([8])
    
def test_double_free(memory_allocator):
    
    blocks_to_allocate = 3
    
    _ = memory_allocator.allocate(blocks_to_allocate)
    
    memory_allocator.free([0])
    
    with pytest.raises(ValueError, match="Block ID 0 is not currently allocated"):
        memory_allocator.free([0])