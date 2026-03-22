import torch # type: ignore
from typing import Tuple

class ContinuousKVCache:
    def __init__(self,
                 batch_size: int,
                 n_layers: int,
                 n_heads: int,
                 head_dim: int,
                 max_seq_len: int,
                 dtype: torch.Tensor,
                 device: str
        ) -> None:
        
        '''
        Description: 
            A class to manage the key and value caches for a transformer model during autoregressive decoding. 
            It initializes the caches, updates them with new key and value tensors, and keeps track of the current sequence length. 
            The cache is organized by layers, heads, and sequence length, allowing for efficient retrieval of past key and value tensors during decoding.
        Args:
            batch_size (int): The batch size for the input sequences.
            n_layers (int): The number of layers in the transformer model.
            n_head (int): The number of attention heads in each layer.
            head_dim (int): The dimensionality of each attention head.
            max_seq_len (int): The maximum sequence length that the cache can accommodate.
            dtype (torch.Tensor): The data type for the key and value tensors in the cache.
            device (str): The device on which to store the cache tensors (e.g., 'cpu' or 'cuda').
        Returns:
            None
        '''
        
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        self.seq_len = [0] * self.batch_size
        
        self.initialize_cache()
        
    def initialize_cache(self) -> None:
        
        '''
        Description:
            Initializes the key and value caches for each layer of the transformer model.
            The caches are created as lists of tensors, where each tensor has the shape 
            (batch_size, n_head, max_seq_len, head_dim) and is initialized to zeros.
        Args:
            None
        Returns:
            None
        '''
        
        self.k_cache = [torch.zeros((self.batch_size, 
                                     self.n_heads, 
                                     self.max_seq_len, 
                                     self.head_dim), 
                                     dtype = self.dtype, 
                                     device = self.device) for _ in range(self.n_layers)]
        
        self.v_cache = [torch.zeros((self.batch_size, 
                                     self.n_heads, 
                                     self.max_seq_len, 
                                     self.head_dim), 
                                     dtype = self.dtype, 
                                     device = self.device) for _ in range(self.n_layers)]
        
    def update_cache(self,
                     layer_idx: int,
                     k: torch.Tensor,
                     v: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        '''
        Description:
            Updates the key and value caches for a specific layer with new key and value tensors. 
            The method takes the layer index, new key and value tensors, and updates the cache for 
            the corresponding batch indices. It also increments the sequence length for the batch indices that have been updated.
        Args:
            layer_idx (int): The index of the layer for which to update the cache.
            k (torch.Tensor): The new key tensor to be added to the cache, with shape (batch_size, n_head, T_new, head_dim).
            v (torch.Tensor): The new value tensor to be added to the cache, with shape (batch_size, n_head, T_new, head_dim).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated key and value caches for the specified layer, 
            with shapes (batch_size, n_head, max_seq_len, head_dim), where max_seq_len is the maximum sequence length after the update.
        '''
        
        for batch_idx in range(self.batch_size):
            seq_len = self.seq_len[batch_idx]
            
            if seq_len < self.max_seq_len:
                start = seq_len
                end = seq_len + k.size(2)
                self.k_cache[layer_idx][batch_idx, :, start: end, :] = k[batch_idx]
                self.v_cache[layer_idx][batch_idx, :, start: end, :] = v[batch_idx]
                
                self.increment_seq_len(batch_idx, layer_idx, k.size(2))
                
        max_seq_len = max(self.seq_len)
        return self.k_cache[layer_idx][ :, :, : max_seq_len, :], \
               self.v_cache[layer_idx][ :, :, : max_seq_len, :]
                
    def increment_seq_len(self, 
                          batch_idx: int,
                          layer_idx: int, 
                          T_new: int
        ) -> None:
        
        '''
        Decription:
            Increments the sequence length for a specific batch index and layer index by the number of new tokens (T_new) 
            that have been added to the cache. 
            This method is called after updating the key and value caches with new tensors, and it ensures that the sequence 
            length is accurately tracked for each batch index and layer index. The sequence length is only incremented for the last layer, 
            as the cache is updated sequentially across layers during decoding.
        Args:
            batch_idx (int): The index of the batch for which to increment the sequence length.
            layer_idx (int): The index of the layer for which to increment the sequence length.
            T_new (int): The number of new tokens that have been added to the cache, which determines how much to increment the sequence length.
        Returns:
            None
        '''
        
        if layer_idx == self.n_layers - 1:
            self.seq_len[batch_idx] += T_new
            
    def reset_slot(self, batch_idx: int) -> None:
        
        '''
        Decription:
            Resets the key and value cache for a specific batch index by setting the corresponding entries in the cache to zero 
            and resetting the sequence length for that batch index to zero. 
            This method is useful when a specific batch index needs to be cleared from the cache, such as when a new sequence is 
            being processed for that batch index or when the cache needs to be cleared for any reason. 
            It ensures that the cache for the specified batch index is cleared and the sequence length is reset, allowing for a 
            fresh start in the decoding process for that batch index.
        Args:
            batch_idx (int): The index of the batch for which to reset the cache slot.
        Returns:
            None
        '''
        
        
        self.seq_len[batch_idx] = 0
        for layer_idx in range(self.n_layers):
            self.k_cache[layer_idx][batch_idx, :, :, :] = 0
            self.v_cache[layer_idx][batch_idx, :, :, :] = 0
            
    def reset_cache(self) -> None:
        
        '''
        Description:
            Resets the key and value caches to their initial state by reinitializing them with zeros and resetting the sequence length to zero. 
            This method is useful when starting a new decoding process or when the cache needs to be cleared for any reason. 
            It ensures that all cached values are cleared and the sequence length is reset, allowing for a fresh start in the decoding process.
        Args:
            None
        Returns:
            None
        '''
        
        self.initialize_cache()
        self.seq_len = [0] * self.batch_size