
'''
Paged KV Cache

Pre-allocates a shared pool of fixed-size blocks on GPU: (num_blocks, n_heads, block_size, head_dim) per layer.
Sequences don't own contiguous memory — their K/V data is scattered across blocks, indexed via BlockTable.
Supports single-token writes, multi-block reads (gather + concat), and per-block reset.

Cache shape per layer: (num_blocks, n_heads, block_size, head_dim)
    - num_blocks: total blocks in the pool (shared across all sequences)
    - block_size: number of token positions per block
'''

import torch

class PagedKVCache:
    def __init__(self,
                 num_blocks: int,
                 n_layers: int,
                 n_heads: int,
                 block_size: int,
                 head_dim: int,
                 dtype: torch.Tensor,
                 device: str,
        ) -> None:
        
        '''
        Description:
            Initializes the paged KV cache with a pool of fixed-size blocks.
            Creates k_cache and v_cache as lists of zero tensors, one per layer,
            each with shape (num_blocks, n_heads, block_size, head_dim).
        Args:
            num_blocks (int): Total number of blocks in the shared pool.
            n_layers (int): Number of transformer layers.
            n_heads (int): Number of attention heads.
            block_size (int): Number of token positions per block.
            head_dim (int): Dimensionality of each attention head.
            dtype (torch.dtype): Data type for cache tensors.
            device (str): Device for cache tensors (e.g., 'cpu' or 'cuda').
        Returns:
            None
        '''
        
        self.num_blocks = num_blocks
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        self.initialize_cache()
        
    def initialize_cache(self):
        
        '''
        Description:
            Allocates k_cache and v_cache as lists of zero tensors.
            Each tensor has shape (num_blocks, n_heads, block_size, head_dim).
            Called once during __init__.
        Args:
            None
        Returns:
            None
        '''
        
        self.k_cache = [torch.zeros((self.num_blocks, 
                                     self.n_heads, 
                                     self.block_size, 
                                     self.head_dim), 
                                    dtype=self.dtype, 
                                    device=self.device) 
                        for _ in range(self.n_layers)]
        
        self.v_cache = [torch.zeros((self.num_blocks, 
                                     self.n_heads, 
                                     self.block_size, 
                                     self.head_dim),
                                    dtype=self.dtype,
                                    device=self.device)
                        for _ in range(self.n_layers)]
        
    def write(self,
              layer_idx: int,
              block_id: int | list[int],
              offset: int | list[int],
              k: torch.Tensor,
              v: torch.Tensor
        ) -> None:
        
        '''
        Description:
            Writes a single token's K/V vectors into the cache at the specified
            block and offset position. Used during decode to store the new token's
            key and value projections.
        Args:
            layer_idx (int): The transformer layer index.
            block_id (int | list[int]): The physical block ID(s) to write into.
            offset (int | list[int]): The token position(s) within the block(s) to write to.
            k (torch.Tensor): Key tensor, shape (n_heads, head_dim).
            v (torch.Tensor): Value tensor, shape (n_heads, head_dim).
        Returns:
            None
        '''
        
        self.k_cache[layer_idx][block_id, :, offset, : ] = k
        self.v_cache[layer_idx][block_id, :, offset, : ] = v
        
    def read(self,
             layer_idx: int,
             block_ids: list[int] | list[list[int]],
             max_blocks: int = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        '''
        Description:
            Reads and concatenates K/V blocks for the specified block IDs. Supports both single sequence (list of block IDs) 
            and batched (list of list of block IDs) modes. For batched mode, shorter sequences are padded with block ID 0, 
            and the caller is responsible for slicing away the padded positions.
        Args:
            layer_idx (int): The transformer layer index.
            block_ids (list[int] | list[list[int]]): Block ID(s) to read. Can be a list of ints for single sequence or a 
            list of list of ints for batched sequences.
            max_blocks (int, optional): Maximum number of blocks per sequence for batched mode. Required if block_ids is a 
            list of list of ints. Used for padding shorter sequences.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (k_cache, v_cache) tensors with shape (B, n_heads, total_tokens, head_dim) 
            for batched mode or (1, n_heads, total_tokens, head_dim) for single sequence mode, where total_tokens = len(block_ids) * block_size 
            for single sequence or max_blocks * block_size for batched mode
        '''
        
        # Single sequence: block_ids = [5, 2, 7]
        if isinstance(block_ids[0], int):
            
            block_ids_t = torch.tensor(block_ids, dtype=torch.long, device=self.device)
            
            # index_select is slightly faster than advanced indexing (self.k_cache[layer_idx][block_ids_t])
            # as it dispatches directly to a dedicated gather kernel, avoiding the
            # general indexing dispatcher's parsing overhead.
            # (len(block_ids), n_heads, block_size, head_dim)
            k_blocks = self.k_cache[layer_idx].index_select(0, block_ids_t)
            v_blocks = self.v_cache[layer_idx].index_select(0, block_ids_t)
        
            # (1, n_heads, len(block_ids)*block_size, head_dim)
            k_cache = k_blocks.permute(1, 0, 2, 3).contiguous().view(self.n_heads, -1, self.head_dim).unsqueeze(0)
            v_cache = v_blocks.permute(1, 0, 2, 3).contiguous().view(self.n_heads, -1, self.head_dim).unsqueeze(0)
            
            return k_cache, v_cache
        
        # Batched: block_ids = [[5, 2, 7], [3, 1], [4, 6, 0, 8]]
        # Left-padded bacthing: block_ids = [[0, 5, 2, 7], [0, 0, 3, 1], [4, 6, 0, 8]]
        B = len(block_ids)
        
        # With left-padded batching, all sequences have equal token counts,
        # so all have the same number of blocks — this padding is a no-op.
        # Kept for generality (e.g., variable-length continuous batching).
        # Pad shorter block ID lists with 0 (reads from block 0 which may hold
        # unrelated data, but these positions are sliced away by the caller).
        padded = [ids + [0] * (max_blocks - len(ids)) for ids in block_ids]
        
        # torch.tensor(padded) creates a 2D (B, max_blocks) tensor directly,
        # then flatten()/view(-1) makes it 1D — cleaner than the nested list comprehension.
        # flatten() works even on non-contiguous tensors (copies if needed),
        # view(-1) requires contiguous memory (guaranteed here since torch.tensor always returns contiguous).
        # flat_ids = torch.tensor([bid for seq in padded for bid in seq], 
        #                         dtype=torch.long, device=self.device)
        flat_ids = torch.tensor(padded, dtype=torch.long, device=self.device).flatten()
        
        # index_select is slightly faster than advanced indexing (self.k_cache[layer_idx][flat_ids])
        # as it dispatches directly to a dedicated gather kernel, avoiding the
        # general indexing dispatcher's parsing overhead.
        # (B*max_blocks, n_heads, block_size, head_dim)
        k_gathered = self.k_cache[layer_idx].index_select(0, flat_ids)
        v_gathered = self.v_cache[layer_idx].index_select(0, flat_ids)
        
        # Reshape to (B, max_blocks, n_heads, block_size, head_dim)
        k_gathered = k_gathered.view(B, max_blocks, self.n_heads, self.block_size, self.head_dim)
        v_gathered = v_gathered.view(B, max_blocks, self.n_heads, self.block_size, self.head_dim)
        
        # (B, n_heads, max_blocks*block_size, head_dim)
        k_cache = k_gathered.permute(0, 2, 1, 3, 4).contiguous().view(B, self.n_heads, -1, self.head_dim)
        v_cache = v_gathered.permute(0, 2, 1, 3, 4).contiguous().view(B, self.n_heads, -1, self.head_dim)
        
        return k_cache, v_cache

    def reset_blocks(self,
                     block_ids: list[int]
        ) -> None:
        
        '''
        Description:
            Zeros out the specified blocks across all layers. Called when a
            sequence finishes and its blocks are returned to the allocator,
            ensuring no stale data remains.
        Args:
            block_ids (list[int]): List of block IDs to zero out.
        Returns:
            None
        '''
        
        block_ids_t = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        for layer_idx in range(self.n_layers):
            self.k_cache[layer_idx][block_ids_t] = 0
            self.v_cache[layer_idx][block_ids_t] = 0