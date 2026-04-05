
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

from .block_table import BlockTable
from .paged_kv_cache import PagedKVCache

class PagedCacheContext:
    def __init__(self,
                paged_kv_cache: PagedKVCache,
                block_table: BlockTable,
                seq_ids: list[str],
                seq_lens: list[int],
                block_size: int,
        ) -> None:
        
        '''
        Description:
            Initializes the adapter with references to the shared paged cache and block table,
            plus per-batch metadata needed to compute block indices and offsets.
        Args:
            paged_kv_cache (PagedKVCache): The shared GPU block pool for K/V storage.
            block_table (BlockTable): Maps sequence IDs to ordered lists of physical block IDs.
            seq_ids (list[str]): Sequence IDs in the current batch, ordered by batch index.
            seq_lens (list[int]): Number of tokens already cached per sequence (before this step).
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
        ) -> tuple[torch.Tensor, torch.Tensor]:
        
        '''
        Description:
            Writes new K/V tokens into paged blocks, then reads back the full history
            for each sequence. Matches the interface of KVCache.update_cache() so the
            attention layer can call it identically via duck typing.

            Write phase: collects all (block_id, offset) pairs across the batch,
                then issues a single batched write to paged_kv_cache.
            Read phase: gathers all block IDs per sequence, calls batched read,
                slices to valid token count.
        Args:
            layer_idx (int): The transformer layer index.
            k (torch.Tensor): New key tensor, shape (B, n_heads, T_new, head_dim).
            v (torch.Tensor): New value tensor, shape (B, n_heads, T_new, head_dim).
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Full cached K and V tensors,
                each with shape (B, n_heads, max_T_total, head_dim).
        '''
        
        batch_size = k.size(0)
        T_new = k.size(2)
        
        # ── WRITE PHASE (vectorized) ──────────────────────────────
        # Compute token positions for all sequences × all new tokens
        # For decode T_new=1; for prefill T_new=prompt_len
        
        all_block_ids = []
        all_offsets = []
        
        for batch_idx in range(batch_size):
            start = self.seq_lens[batch_idx]
            # For decode stage with seq_len = 1, we still want to write the 
            # single new token into the cache and read back the full history.
            for t in range(T_new):
                
                token_pos = start + t
                block_index = token_pos // self.block_size
                offset = token_pos % self.block_size
                
                # Auto-allocate if needed 
                if block_index >= self.block_table.num_blocks(self.seq_ids[batch_idx]):
                    self.block_table.allocate_blocks(self.seq_ids[batch_idx], 1)
                    
                block_id = self.block_table.get_physical_block(seq_id = self.seq_ids[batch_idx],
                                                               logical_index = block_index)
                
                all_block_ids.append(block_id)
                all_offsets.append(offset)
                
        # Reshape k, v from (B, n_heads, T_new, head_dim) to (B*T_new, n_heads, head_dim)
        k_reshaped = k.permute(0, 2, 1, 3).reshape(-1, k.size(1), k.size(3))
        v_reshaped = v.permute(0, 2, 1, 3).reshape(-1, v.size(1), v.size(3))
        
        # Single batched write — one GPU kernel instead of B*T_new calls
        self.paged_kv_cache.write(layer_idx = layer_idx,
                                  block_id = all_block_ids,
                                  offset = all_offsets,
                                  k = k_reshaped,
                                  v = v_reshaped)
        
        # ── READ PHASE (batched) ──────────────────────────────────
        # Gather all block IDs for all sequences
        
        all_seq_block_ids, max_blocks = self.block_table.get_block_ids_for_batch(self.seq_ids)
        
        # Single batched read — one index_select instead of B
        k_full, v_full = self.paged_kv_cache.read(layer_idx=layer_idx,
                                                block_ids=all_seq_block_ids,
                                                max_blocks=max_blocks)
        # shape: (B, n_heads, max_blocks * block_size, head_dim)
        
        # Slice away trailing block padding — keep only valid tokens
        max_valid = max(self.seq_lens[i] + T_new for i in range(batch_size))
        k_full = k_full[:, :, :max_valid, :]
        v_full = v_full[:, :, :max_valid, :]
        
        # ── INCREMENT SEQ LENS ────────────────────────────────
        self.increment_seq_len(layer_idx, T_new)
        
        return k_full, v_full
                
    def increment_seq_len(self, 
                          layer_idx: int, 
                          T_new: int
        ) -> None:
        
        '''
        Description:
            Increments the token count for each sequence after writing new tokens to the cache.
             Only updates after the last layer to avoid double counting when multiple layers
             write to the same cache.
        Args:
            layer_idx (int): The transformer layer index.
            T_new (int): Number of new tokens added in this step.
        Returns:
            None
        '''
        
        if layer_idx == self.paged_kv_cache.n_layers - 1:
            self.seq_lens = [s + T_new for s in self.seq_lens]
            
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
        