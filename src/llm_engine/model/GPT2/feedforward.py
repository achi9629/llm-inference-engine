import torch
import torch.nn as nn
from typing import Callable
from transformers.activations import gelu_new, gelu

from .conv import Conv1D

class FeedForwardNetwork(nn.Module):
    def __init__(self, 
                 n_embd: int, 
                 n_inner: int, 
                 activation_function: str = 'gelu_new'
        ) -> None:
        super(FeedForwardNetwork, self).__init__()
        
        '''
        Description:
            FFN module as described in the GPT-2 paper. It consists of two linear layers with an activation function in between.
        Arguments:
            n_embd: the dimensionality of the input and output embeddings
            n_inner: the dimensionality of the inner layer of the feedforward network
            activation_function: the activation function to use in the feedforward network 
                                (default: 'gelu_new', options: 'gelu_new', 'gelu')
        Returns:
            None
        '''
        
        if n_inner <= n_embd:
            raise ValueError(f"n_inner must be greater than n_embd. Got n_inner={n_inner}, n_embd={n_embd}")
        
        self.c_fc   = Conv1D(n_inner, n_embd)
        self.c_proj = Conv1D(n_embd, n_inner)
        
        # Activation function takes a tensor and returns a tensor of the same shape.
        self.c_act: Callable[[torch.Tensor], torch.Tensor]
        if activation_function == 'gelu_new':
            self.c_act = gelu_new
        elif activation_function == 'gelu':
            self.c_act = gelu
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        '''
        Arguments:
            x: the input tensor of shape (batch_size, seq_length, n_embd)
        Returns:
            the output tensor of shape (batch_size, seq_length, n_embd)
        '''
        
        h = self.c_fc(x)     # (batch_size, seq_length, n_embd -> batch_size, seq_length, n_inner)
        h = self.c_act(h)    # (batch_size, seq_length, n_inner -> batch_size, seq_length, n_inner)
        h = self.c_proj(h)   # (batch_size, seq_length, n_inner -> batch_size, seq_length, n_embd)
        return h
    