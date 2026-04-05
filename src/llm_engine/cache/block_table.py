
'''
Block Table for Paged KV Cache

Maps sequence IDs to their ordered list of physical block IDs.
Acts as the "page table" — translates logical token positions to physical block locations.
Delegates allocation/deallocation to the MemoryAllocator.

Logical-to-physical mapping:
    block_index = token_position // block_size
    offset      = token_position % block_size
    block_id    = block_table[seq_id][block_index]
'''

from .memory_allocator import MemoryAllocator

class BlockTable:
    def __init__(self, 
                 allocator: MemoryAllocator, 
                 block_size: int
        ) -> None:
        
        '''
        Description:
            Initializes the block table with a reference to the memory allocator
            and the fixed block size. Creates an empty table (dict) to map sequence IDs
            to their ordered list of block IDs.
        Args:
            allocator (MemoryAllocator): The memory allocator instance for block allocation/deallocation.
            block_size (int): Number of token positions per block.
        Returns:
            None
        '''
        
        self.allocator = allocator
        self.block_size = block_size
        self.block_table = {}
        
    def add_sequence(self, 
                     seq_id: str
        ) -> None:
        
        '''
        Description:
            Registers a new sequence in the block table with an empty block list.
            Must be called before allocating blocks for a sequence.
        Args:
            seq_id (str): Unique identifier for the sequence.
        Returns:
            None
        Raises:
            ValueError: If seq_id already exists in the block table.
        '''
        
        if seq_id in self.block_table:
            raise ValueError(f"Sequence ID {seq_id} already exists in the block table.")
        
        self.block_table[seq_id] = []
        
    def allocate_blocks(self, 
                        seq_id: str, 
                        num_blocks: int
        ) -> list[int]:
        
        '''
        Description:
            Allocates num_blocks from the memory allocator and appends them
            to the sequence's block list. Called during prefill or when
            the current last block is full and more space is needed.
        Args:
            seq_id (str): Unique identifier for the sequence.
            num_blocks (int): Number of new blocks to allocate.
        Returns:
            list[int]: The newly allocated block IDs.
        Raises:
            ValueError: If seq_id does not exist in the block table.
            MemoryError: If the allocator has insufficient free blocks.
        '''
        
        if seq_id not in self.block_table:
            raise ValueError(f"Sequence ID {seq_id} does not exist in the block table.")
        
        block_ids = self.allocator.allocate(num_blocks)
        
        self.block_table[seq_id].extend(block_ids)
        
        return block_ids
        
    def get_block_ids(self, 
                      seq_id: str
        ) -> list[int]:
        
        '''
        Description:
            Returns the ordered list of block IDs assigned to a sequence.
            Used during attention to gather all K/V data for a sequence.
        Args:
            seq_id (str): Unique identifier for the sequence.
        Returns:
            list[int]: Ordered block IDs for the sequence.
        Raises:
            ValueError: If seq_id does not exist in the block table.
        '''
        
        if seq_id not in self.block_table:
            raise ValueError(f"Sequence ID {seq_id} does not exist in the block table.")
        
        return self.block_table[seq_id]
    
    def get_block_ids_for_batch(self, 
                               seq_ids: list[str],
        ) -> tuple[list[list[int]], int]:
        
        '''
        Description:
            Returns the ordered list of block IDs for each sequence in a batch,
            along with the maximum number of blocks allocated to any sequence in the batch.
            Used during attention to gather K/V data for a batch of sequences.
        Args:
            seq_ids (list[str]): List of unique identifiers for the sequences in the batch.
        Returns:
            tuple[list[list[int]], int]: A tuple containing:
                - A list of lists of block IDs, one per sequence.
                - The maximum number of blocks allocated to any sequence in the batch.
        Raises:
            ValueError: If any seq_id does not exist in the block table.
        '''
        
        if any(seq_id not in self.block_table for seq_id in seq_ids):
            missing_ids = [seq_id for seq_id in seq_ids if seq_id not in self.block_table]
            raise ValueError(f"Sequence IDs {missing_ids} do not exist in the block table.")
        
        block_id_lists = [self.block_table[seq_id] for seq_id in seq_ids]
        max_blocks = max(len(block_ids) for block_ids in block_id_lists)
        return block_id_lists, max_blocks
    
    def get_physical_block(self, 
                           seq_id: str, 
                           logical_index: int
        ) -> int:
        
        '''
        Description:
            Returns the physical block ID at a given logical index in the
            sequence's block list. Used to locate where a specific token's
            K/V should be written: block_id = get_physical_block(seq_id, token_pos // block_size).
        Args:
            seq_id (str): Unique identifier for the sequence.
            logical_index (int): Index into the sequence's block list.
        Returns:
            int: The physical block ID.
        Raises:
            ValueError: If seq_id does not exist in the block table.
            ValueError: If logical_index is out of bounds.
        '''
        
        if seq_id not in self.block_table:
            raise ValueError(f"Sequence ID {seq_id} does not exist in the block table.")
        
        if logical_index < 0 or logical_index >= len(self.block_table[seq_id]):
            raise ValueError(f"Logical index {logical_index} is out of bounds for sequence ID {seq_id}.")
        
        return self.block_table[seq_id][logical_index]
    
    def get_physical_blocks_for_batch(self,
                                      seq_ids: list[int],
                                      logical_indices: list[int]
        ) -> list[int]:
        
        '''
        Description:
            Returns the physical block IDs for a batch of sequences at their respective logical indices.
            Used during attention to gather the correct K/V blocks for each sequence in the batch.
        Args:
            seq_ids (list[int]): List of unique identifiers for the sequences in the batch.
            logical_indices (list[int]): List of logical indices corresponding to each sequence.
        Returns:
            list[int]: List of physical block IDs corresponding to each sequence and logical index.
        Raises:
            ValueError: If any seq_id does not exist in the block table.
            ValueError: If any logical_index is out of bounds for its respective sequence.
        '''
        
        if any(seq_id not in self.block_table for seq_id in seq_ids):
            missing_ids = [seq_id for seq_id in seq_ids if seq_id not in self.block_table]
            raise ValueError(f"Sequence IDs {missing_ids} do not exist in the block table.")
        
        if any(logical_index < 0 or logical_index >= len(self.block_table[seq_id]) 
               for seq_id, logical_index in zip(seq_ids, logical_indices)):
            out_of_bounds = [(seq_id, logical_index) for seq_id, logical_index in zip(seq_ids, logical_indices) 
                             if logical_index < 0 or logical_index >= len(self.block_table[seq_id])]
            raise ValueError(f"Logical indices {out_of_bounds} are out of bounds for their respective sequence IDs.")
        
        return [self.get_physical_block(seq_id, logical_index) for seq_id, logical_index 
                in zip(seq_ids, logical_indices)]
    
    def free_sequence(self, 
                      seq_id: str
        ) -> None:
        
        '''
        Description:
            Frees all blocks allocated to a sequence by returning them to the
            memory allocator, then removes the sequence from the block table.
            Called when a sequence finishes generation.
        Args:
            seq_id (str): Unique identifier for the sequence.
        Returns:
            None
        Raises:
            ValueError: If seq_id does not exist in the block table.
        '''
        
        if seq_id not in self.block_table:
            raise ValueError(f"Sequence ID {seq_id} does not exist in the block table.")
        
        self.allocator.free(self.block_table[seq_id])
        del self.block_table[seq_id]
        
    def num_blocks(self, 
                   seq_id: str
        ) -> int:
        
        '''
        Description:
            Returns the number of blocks currently allocated to a sequence.
        Args:
            seq_id (str): Unique identifier for the sequence.
        Returns:
            int: Number of blocks allocated to the sequence.
        Raises:
            ValueError: If seq_id does not exist in the block table.
        '''
        
        if seq_id not in self.block_table:
            raise ValueError(f"Sequence ID {seq_id} does not exist in the block table.")
        
        return len(self.block_table[seq_id])
        
