
'''
Memory Allocator for Paged KV Cache

Manages a pool of fixed-size block IDs using a deque (free list) + set (allocated tracking).
Agnostic to K/V tensors, layers, or attention — only tracks block IDs.
Used by PagedKVCache (Day 16) to allocate/free blocks on demand.

Time Complexity: O(k) allocate, O(m) free, O(1) num_free_blocks
'''

from typing import List
from collections import deque

class MemoryAllocator:
    def __init__(self, num_blocks: int) -> None:
        
        '''
        Description:
            Initializes the memory allocator with a fixed number of blocks.
            Creates a FIFO free list (deque) containing all block IDs [0, num_blocks-1]
            and an empty allocated set for O(1) membership tracking.
        Args:
            num_blocks (int): Total number of blocks in the pool.
        Returns:
            None
        '''
    
        self.num_blocks = num_blocks
        self.free_blocks = deque(range(num_blocks))
        self.allocated_blocks = set()
        
    def allocate(self, num_blocks: int) -> List[int]:
        
        '''
        Description:
            Pops num_blocks block IDs from the front of the free list and adds them
            to the allocated set. Raises MemoryError if insufficient blocks available.
        Args:
            num_blocks (int): Number of blocks to allocate.
        Returns:
            List[int]: List of allocated block IDs.
        Raises:
            MemoryError: If num_blocks exceeds the number of free blocks.
        '''
        
        if num_blocks > len(self.free_blocks):
            raise MemoryError("Not enough free blocks available, Needed: {}, Available: {}and Allocated: {}".format(num_blocks, len(self.free_blocks), len(self.allocated_blocks)))
        
        allocated = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.popleft()
            self.allocated_blocks.add(block_id)
            allocated.append(block_id)
            
        return allocated
    
    def free(self, block_ids: List[int]) -> None:
        
        '''
        Description:
            Returns block IDs to the free list and removes them from the allocated set.
            Validates each block ID for bounds and double-free before releasing.
        Args:
            block_ids (List[int]): List of block IDs to free.
        Returns:
            None
        Raises:
            ValueError: If a block ID is out of range [0, num_blocks-1].
            ValueError: If a block ID is not currently allocated (double-free).
        '''
        
        for block_id in block_ids:
            if block_id < 0 or block_id >= self.num_blocks:
                raise ValueError(f"Invalid block ID: {block_id}")
            if block_id not in self.allocated_blocks:
                raise ValueError(f"Block ID {block_id} is not currently allocated")
            self.allocated_blocks.remove(block_id)
            self.free_blocks.append(block_id)
        
    @property
    def num_free_blocks(self) -> int:
        
        '''
        Description:
            Returns the number of blocks currently available for allocation.
        Args:
            None
        Returns:
            int: Count of free blocks.
        '''
        
        return len(self.free_blocks)