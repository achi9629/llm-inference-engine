
'''
Paged Cache Context (Adapter)

Wraps PagedKVCache + BlockTable behind the same update_cache(layer_idx, k, v) interface
that KVCache and ContinuousKVCache expose. This lets the attention layer use paged memory
without any code changes — duck typing handles the dispatch.

Internally, update_cache() loops over each sequence in the batch:
  1. Write: stores each new token's K/V into the correct (block_id, offset) slot
  2. Read: gathers all blocks for the sequence, slices to valid tokens only
  3. Stack: pads and batches per-sequence results into (B, n_heads, max_T_total, head_dim)

Also tracks per-sequence token counts (seq_lens) and exposes a seq_len property
for position embedding offset in transformer.py.
'''

import torch
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence

from .block_table import BlockTable
from .paged_kv_cache import PagedKVCache

class PagedCacheContext:
    def __init__(self,
                paged_kv_cache: PagedKVCache,
                block_table: BlockTable,
                seq_ids: List[str],
                seq_lens: List[int],
                block_size: int,
        ) -> None:
        
        '''
        Description:
            Initializes the adapter with references to the shared paged cache and block table,
            plus per-batch metadata needed to compute block indices and offsets.
        Args:
            paged_kv_cache (PagedKVCache): The shared GPU block pool for K/V storage.
            block_table (BlockTable): Maps sequence IDs to ordered lists of physical block IDs.
            seq_ids (List[str]): Sequence IDs in the current batch, ordered by batch index.
            seq_lens (List[int]): Number of tokens already cached per sequence (before this step).
            block_size (int): Number of token slots per block.
        Returns:
            None
        '''
        
        self.paged_kv_cache = paged_kv_cache
        self.block_table = block_table
        self.seq_ids = seq_ids
        self.seq_lens = seq_lens
        self.block_size = block_size
        
    def update_cache(self,
                     layer_idx: int,
                     k: torch.Tensor,
                     v: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        '''
        Description:
            Writes new K/V tokens into paged blocks, then reads back the full history
            for each sequence. Matches the interface of KVCache.update_cache() so the
            attention layer can call it identically via duck typing.

            For each sequence in the batch:
            - Write phase: loops over T_new tokens, computes (block_id, offset) from
                token position, writes to paged_kv_cache.
            - If a token's block doesn't exist yet, allocates it on the fly.
            - Read phase: gathers all blocks, slices to valid tokens (0..start+T_new),
                discarding trailing zeros from partially filled blocks.
            - Pads shorter sequences to match the longest, producing a batch tensor.
        Args:
            layer_idx (int): The transformer layer index.
            k (torch.Tensor): New key tensor, shape (B, n_heads, T_new, head_dim).
            v (torch.Tensor): New value tensor, shape (B, n_heads, T_new, head_dim).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Full cached K and V tensors,
                each with shape (B, n_heads, max_T_total, head_dim).
        '''
        
        k_full = []
        v_full = []
        
        batch_size = k.size(0)
        for batch_idx in range(batch_size):
            start = self.seq_lens[batch_idx]
            end = k.size(2)
            for t in range(end):
                token_pos = start + t
                block_index = token_pos // self.block_size
                offset = token_pos % self.block_size
                
                if block_index >= self.block_table.num_blocks(self.seq_ids[batch_idx]):
                    self.block_table.allocate_blocks(self.seq_ids[batch_idx], 1)
                
                block_id = self.block_table.get_physical_block(seq_id = self.seq_ids[batch_idx],
                                                               logical_index = block_index)
                self.paged_kv_cache.write(layer_idx = layer_idx,
                                          block_id = block_id,
                                          offset = offset,
                                          k = k[batch_idx, :, t, :],
                                          v = v[batch_idx, :, t, :])
                
            block_ids = self.block_table.get_block_ids(seq_id = self.seq_ids[batch_idx])
            
            k_seq, v_seq = self.paged_kv_cache.read(layer_idx =layer_idx,
                                                    block_ids = block_ids)
            
            k_seq = k_seq[ :, :, : start + end, : ].squeeze(0)
            v_seq = v_seq[ :, :, : start + end, : ].squeeze(0)
            
            k_full.append(k_seq.transpose(0,1))
            v_full.append(v_seq.transpose(0,1))
            
            self.increment_seq_len(batch_idx = batch_idx,
                                   layer_idx = layer_idx,
                                   T_new = end)
            
        k_full = pad_sequence(k_full, batch_first=True, padding_value=0.0).transpose(1,2)
        v_full = pad_sequence(v_full, batch_first=True, padding_value=0.0).transpose(1,2)
        
        return k_full, v_full
            
    def increment_seq_len(self, 
                          batch_idx: int, 
                          layer_idx: int, 
                          T_new: int
        ) -> None:
        
        '''
        Description:
            Increments a sequence's cached token count by T_new. Only fires on the
            last layer to prevent double-counting — all layers see the same seq_lens
            during a single forward pass, and the count updates once at the end.
        Args:
            batch_idx (int): Index of the sequence in the current batch.
            layer_idx (int): The transformer layer index.
            T_new (int): Number of new tokens added in this step.
        Returns:
            None
        '''
        
        if layer_idx == self.paged_kv_cache.n_layers - 1:
            self.seq_lens[batch_idx] += T_new
            
    @property
    def seq_len(self) -> int:
        
        '''
        Description:
            Returns the maximum sequence length across all sequences in the batch.
            Used by transformer.py to compute position embedding offsets when
            padding_mask is None.
        '''

        return max(self.seq_lens)
            
    def reset_blocks(self) -> None:
        
        '''
        Description:
            Resets all blocks used by the current batch's sequences, clearing their
            contents and making them available for reuse. Should be called at the end
            of each forward pass to prevent stale data from being read in the next pass.
        Args:
            None
        Returns:
            None
        '''
        
        for seq_id in self.seq_ids:
            block_id = self.block_table.get_block_ids(seq_id = seq_id)
            self.paged_kv_cache.reset_blocks(block_ids = block_id)
        