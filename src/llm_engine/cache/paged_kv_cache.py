
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
from typing import List, Tuple

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
              block_id: int,
              offset: int,
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
            block_id (int): The physical block ID (from BlockTable).
            offset (int): The position within the block (token_pos % block_size).
            k (torch.Tensor): Key tensor, shape (n_heads, head_dim).
            v (torch.Tensor): Value tensor, shape (n_heads, head_dim).
        Returns:
            None
        '''
        
        self.k_cache[layer_idx][block_id, :, offset, : ] = k
        self.v_cache[layer_idx][block_id, :, offset, : ] = v
        
    def read(self,
             layer_idx: int,
             block_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        '''
        Description:
            Gathers K/V data from multiple scattered blocks and concatenates them
            into contiguous tensors. Used during attention to reconstruct a
            sequence's full K/V history from its block list.
        Args:
            layer_idx (int): The transformer layer index.
            block_ids (List[int]): Ordered list of physical block IDs for a sequence.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Concatenated K and V tensors,
                each with shape (1, n_heads, len(block_ids) * block_size, head_dim).
        '''
        
        k_cache = torch.cat([self.k_cache[layer_idx][block_id, :, :, : ] 
                             for block_id in block_ids], dim=1).unsqueeze(0)
        v_cache = torch.cat([self.v_cache[layer_idx][block_id, :, :, : ] 
                             for block_id in block_ids], dim=1).unsqueeze(0)
        
        return k_cache, v_cache
    
    def reset_blocks(self,
                     block_ids: List[int]
        ) -> None:
        
        '''
        Description:
            Zeros out the specified blocks across all layers. Called when a
            sequence finishes and its blocks are returned to the allocator,
            ensuring no stale data remains.
        Args:
            block_ids (List[int]): List of block IDs to zero out.
        Returns:
            None
        '''
        
        for block_id in block_ids:
            for layer_idx in range(self.n_layers):
                self.k_cache[layer_idx][block_id, :, :, : ] = 0
                self.v_cache[layer_idx][block_id, :, :, : ] = 0