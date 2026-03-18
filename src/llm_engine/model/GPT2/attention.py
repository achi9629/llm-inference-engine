import torch # type: ignore
import torch.nn as nn # type: ignore

from .conv import Conv1D

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 n_embd: int,
                 n_heads: int,
                 n_ctx: int,
                 attn_pdrop: float = 0.0,
        ) -> None:
        super(MultiHeadAttention, self).__init__()
        
        '''
        Description:
            Multi-Head Attention module for transformer-based models.
        Arguments:
            n_embd: the dimensionality of the input and output vectors
            n_head: the number of attention heads
            n_ctx: the maximum context length (sequence length)
            attn_pdrop: the dropout probability for the attention weights
        Returns:
            None
        '''
        if n_embd % n_heads != 0:
            raise ValueError(f"Embedding dimension {n_embd} must be divisible by the number of heads {n_heads}")
        
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.n_ctx  = n_ctx
        
        self.c_attn = Conv1D(3 * self.n_embd, self.n_embd)
        self.c_proj = Conv1D(self.n_embd, self.n_embd)
        
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.scale = (self.n_embd // self.n_heads) ** -0.5
        
        # Causal mask to ensure that attention is only applied to previous tokens in the sequence
        self.register_buffer("bias", torch.tril(torch.ones(self.n_ctx, self.n_ctx)).view(1, 1, self.n_ctx, self.n_ctx))
        
    def forward(self, 
                x: torch.Tensor,
                layer_idx: int = None,
                kv_cache: object = None,
                padding_mask: torch.Tensor = None
        ) -> torch.Tensor:
        
        '''
        Description:
            Forward pass for the Multi-Head Attention module. 
            This method computes the attention output for a given input tensor, 
            optionally using cached key and value tensors from previous time steps (useful during autoregressive decoding).
            options for handling padding masks can also be included to prevent attention to padded tokens.
            
        Arguments:
            x: input tensor of shape (batch_size, sequence_length, embedding_dim)
            layer_idx: an optional index of the current layer (used for caching key and value tensors)
            kv_cache: an optional object for caching key and value tensors across time steps (used during autoregressive decoding)
            padding_mask: an optional tensor of shape (batch_size, sequence_length) where 0 indicates positions 
                that should be masked (not attended to) and 1 indicates valid positions
            
        Returns:
            output tensor of shape (batch_size, sequence_length, embedding_dim)
        '''
        
        B, T_new, C = x.size()
        device = x.device
        
        # Compute query, key, and value projections
        qkv = self.c_attn(x).view(B, T_new, 3, self.n_heads, C // self.n_heads)  # (B, T_new, 3 * C) -> (B, T_new, 3, n_head, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                                       # (3, B, n_head, T_new, head_dim)
        q,k,v = qkv[0], qkv[1], qkv[2]                                         # each is (B, n_head, T_new, head_dim)
        
        if kv_cache is not None and layer_idx is not None:
            # Update the key and value tensors with the cached values from previous time steps.
            # Return the updated key and value tensors for use in the attention computation.
            k, v = kv_cache.update_cache(layer_idx, k, v)
            
            # Total sequence length including cached tokens
            T_total = k.size(2)
        else:
            T_total = T_new
        
        score = ( q @ k.transpose(-2, -1) ) * self.scale                  # (B, n_head, T_new, head_dim) @ (B, n_head, head_dim, T_new) -> (B, n_head, T_new, T_total)
        
        if self.bias.size(2) < T_new or self.bias.size(3) < T_total:
            raise ValueError(f"Causal mask size {self.bias.size()} is smaller than required for sequence length {T_new} and total length {T_total}")
        
        causal_mask = self.bias[:, :, T_total - T_new: T_total, : T_total].to(device)
        # torch.finfo(score.dtype).min gives the minimum representable value for the data type of score, 
        # which is used to effectively mask out positions in the attention scores that should not be attended 
        # to (e.g., future tokens in causal attention).
        score = score.masked_fill(causal_mask == 0, torch.finfo(score.dtype).min)
        
        if padding_mask is not None:
            # Expand the padding mask to match the dimensions of the attention scores
            if kv_cache is not None and layer_idx is not None:
                # If using kv_cache, we need to account for the total sequence length (including cached tokens)
                padding_mask = padding_mask[:, :T_total]  # Ensure padding mask matches the total sequence length
            
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_total)
            
            # torch.finfo(score.dtype).min gives the minimum representable value for the data type of score, 
            # which is used to effectively mask out positions in the attention scores that should not be attended 
            # to (e.g., future tokens in causal attention).
            score = score.masked_fill(padding_mask == 0, torch.finfo(score.dtype).min)
        
        attn_weights = torch.softmax(score, dim=-1)                   # (B, n_head, T_new, T_total)
        attn_weights = self.attn_drop(attn_weights)
        
        out = attn_weights @ v                                        # (B, n_head, T_new, T_total) @ (B, n_head, T_total, head_dim) -> (B, n_head, T_new, head_dim)
        
        out = out.transpose(1, 2).contiguous().view(B, T_new, C)      # (B, n_head, T_new, head_dim) -> (B, T_new, n_head * head_dim) -> (B, T_new, C)
        out = self.c_proj(out)                                        # (B, T_new, C) -> (B, T_new, C)
        
        return out

        

                
        