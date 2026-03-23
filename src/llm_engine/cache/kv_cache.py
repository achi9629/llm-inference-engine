import torch
from typing import Tuple

class KVCache:
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
        '''
        
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        
        self.initialize_cache()
        self.seq_len = 0
        
    def initialize_cache(self) -> None:
        
        '''
        Description:
            Initializes the key and value caches for each layer of the transformer model.
            The caches are created as lists of tensors, where each tensor has the shape 
            (batch_size, n_head, max_seq_len, head_dim) and is initialized to zeros.
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
            The new tensors are added to the cache at the appropriate position based on the current sequence length, and the sequence length is incremented accordingly. 
            The method returns the updated key and value tensors for the layer, which include all the cached values up to the current sequence length.
        
        Args:
            layer_idx (int): The index of the layer for which to update the cache.
            k (torch.Tensor): The new key tensor to be added to the cache, with shape (batch_size, n_head, T_new, head_dim).
            v (torch.Tensor): The new value tensor to be added to the cache, with shape (batch_size, n_head, T_new, head_dim).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated key and value tensors for the specified layer, 
            with shapes (batch_size, n_head, T_total, head_dim), where T_total is the total sequence length after the update.
        '''
        
        start = self.seq_len
        end   = start + k.shape[2]
        
        self.k_cache[layer_idx][:, :, start: end, :] = k
        self.v_cache[layer_idx][:, :, start: end, :] = v
        
        self.increment_seq_len(layer_idx, k.shape[2])
        
        return self.k_cache[layer_idx][:, :, : end, :], self.v_cache[layer_idx][:, :, : end, :]
    
    def increment_seq_len(self, 
                          layer_idx: int, 
                          T_new: int
        ) -> None:
        
        '''
        Description:
            Increments the current sequence length by the number of new tokens added to the cache. 
            This method is called after updating the cache with new key and value tensors, 
            and it ensures that the sequence length is accurately tracked for each layer.
            The sequence length is only updated for the last layer, as it represents the total 
            length of the input sequence processed so far.
        '''
        
        if layer_idx  == self.n_layers - 1:
            self.seq_len += T_new
    
    def reset_cache(self) -> None:
        
        '''
        Description:
            Resets the key and value caches to their initial state by reinitializing them with zeros and resetting the sequence length to zero. 
            This method is useful when starting a new decoding process or when the cache needs to be cleared for any reason. 
            It ensures that all cached values are cleared and the sequence length is reset, allowing for a fresh start in the decoding process.
        '''

        self.seq_len = 0
        self.initialize_cache()