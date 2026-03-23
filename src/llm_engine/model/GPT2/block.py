import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForwardNetwork

class TransformerBlock(nn.Module):
    def __init__(self,
                 n_embd: int,
                 n_heads: int,
                 n_ctx: int,
                 n_inner: int,
                 attn_pdrop: float,
                 resid_pdrop: float,
                 activation_function: str,
                 layer_norm_epsilon: float,
        ) -> None:
        
        '''
        Description:
            A single transformer block that consists of a multi-head self-attention layer followed by a feed
        Args:
            n_embd: The dimensionality of the embeddings and hidden states.
            n_head: The number of attention heads.
            n_ctx: The maximum context length (sequence length).
            n_inner: The dimensionality of the inner feedforward layer.
            attn_pdrop: The dropout probability for the attention weights.
            activation_function: The activation function to use in the feedforward network (e.g., 'relu', 'gelu').
            resid_pdrop: The dropout probability for the residual connections.
            layer_norm_epsilon: The epsilon value for layer normalization to prevent division by zero.
        Returns:
            None
        '''
        
        super(TransformerBlock, self).__init__()
        
        self.ln_1 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon, bias=True)
        self.attn = MultiHeadAttention(n_embd, n_heads, n_ctx, attn_pdrop)
        
        self.ln_2 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon, bias=True)
        self.mlp  = FeedForwardNetwork(n_embd, n_inner, activation_function)
        
        self.resid_drop = nn.Dropout(resid_pdrop)
        
    def forward(self, 
                x: torch.Tensor,
                layer_idx: int = None,
                kv_cache: object = None,
                padding_mask: torch.Tensor = None
        ) -> torch.Tensor:
        
        '''
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd)
            layer_idx: The index of the current transformer block layer (used for caching in attention).
            kv_cache: An optional cache object for storing key and value tensors in attention to speed up decoding.
            padding_mask: an optional tensor of shape (batch_size, sequence_length) where 0 indicates positions 
                that should be masked (not attended to) and 1 indicates valid positions
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_embd)
        '''
        
        # MHA sub-layer with residual connection and layer normalization
        mha = self.attn(self.ln_1(x), layer_idx = layer_idx, kv_cache = kv_cache, padding_mask = padding_mask)
        x = x + self.resid_drop(mha)
        
        # FFN sub-layer with residual connection and layer normalization
        ffn = self.mlp(self.ln_2(x))
        x = x + self.resid_drop(ffn)
        
        return x