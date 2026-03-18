import torch # type: ignore
import torch.nn as nn # type: ignore

from .block import TransformerBlock

class GPT2(nn.Module):
    def __init__(self, model_cfg):
        super(GPT2, self).__init__()
        
        '''
        Description:
            GPT-2 language model: token/position embeddings, N transformer blocks, final LN,
            and tied LM head (wte.weight reused for logit projection).
        
        Parameters:
            model_cfg (dict): A dictionary containing the configuration parameters for the GPT-2 model, including:
                - vocab_size (int): The size of the vocabulary.
                - n_positions (int): The maximum number of positions (sequence length) the model can handle.
                - n_ctx (int): The context size for the model.  
                - n_embd (int): The dimensionality of the embeddings and hidden states.
                - n_layer (int): The number of transformer blocks in the model.
                - n_head (int): The number of attention heads in each transformer block.
                - n_inner (int): The dimensionality of the inner feed-forward layer in each transformer block.
                - activation_function (str): The activation function to use in the feed-forward layers (e.g., 'relu', 'gelu').
                - resid_pdrop (float): The dropout probability for the residual connections in the transformer blocks.
                - embd_pdrop (float): The dropout probability for the embedding layer.
                - attn_pdrop (float): The dropout probability for the attention weights in the transformer blocks.
                - layer_norm_epsilon (float): The epsilon value for layer normalization to prevent division by zero.
        
        Returns:
            None
        '''
        
        self.vocab_size = model_cfg['vocab_size']
        self.n_positions = model_cfg['n_positions']
        self.n_ctx = model_cfg['n_ctx']
        self.n_embd = model_cfg['n_embd']
        self.n_layer = model_cfg['n_layer']
        self.n_heads = model_cfg['n_heads']
        self.n_inner = model_cfg['n_inner']
        self.activation_function = model_cfg['activation_function']
        self.resid_pdrop = model_cfg['resid_pdrop']
        self.embd_pdrop = model_cfg['embd_pdrop']
        self.attn_pdrop = model_cfg['attn_pdrop']
        self.layer_norm_epsilon = model_cfg['layer_norm_epsilon']
        
        self.wte = nn.Embedding(self.vocab_size, self.n_embd)
        self.wpe = nn.Embedding(self.n_positions, self.n_embd)
        
        self.h = nn.ModuleList([TransformerBlock(self.n_embd,
                                                 self.n_heads,
                                                 self.n_ctx,
                                                 self.n_inner,
                                                 self.attn_pdrop,
                                                 self.resid_pdrop,
                                                 self.activation_function,
                                                 self.layer_norm_epsilon
                                                 ) 
                                for _ in range(self.n_layer)])
        
        self.embd_drop = nn.Dropout(self.embd_pdrop)
        self.ln_f = nn.LayerNorm(self.n_embd, eps=self.layer_norm_epsilon, bias=True)
        
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
        
    def forward(self, 
                input_ids: torch.Tensor,
                kv_cache: object = None,
                padding_mask: torch.Tensor = None,
            ) -> torch.Tensor:
        
        '''
        Description:
            Full forward pass: embed tokens + positions, apply transformer blocks,
            layer-norm, and project to vocab logits via tied wte weights.
            
            When kv_cache is provided, position embeddings are offset by the
            number of previously cached tokens (kv_cache.seq_len), so that
            decode-phase tokens receive correct absolute positions.
        
        Args:
            input_ids (torch.Tensor): Token IDs of shape (B, T), dtype torch.long.
                During prefill, T is the full prompt length.
                During decode with kv_cache, T is 1 (single new token).
            kv_cache (object, optional): KVCache instance for reusing past K/V
                tensors across decoding steps. None disables caching.
            padding_mask: (torch.Tensor, optional): an optional tensor of shape (batch_size, sequence_length) where 0 indicates positions 
                that should be masked (not attended to) and 1 indicates valid positions
            
        Returns:
            torch.Tensor: Logits of shape (B, T, vocab_size).
        '''
        
        if input_ids.dtype != torch.long:
            raise ValueError(f"Input tensor must be of type torch.long, but got {input_ids.dtype}")
        
        if input_ids.ndim != 2:
            raise ValueError(f"Input tensor must be 2-dimensional (B, T), but got {input_ids.ndim} dimensions")
        
        _, T = input_ids.size()
        
        if T > self.n_ctx:
            raise ValueError(f"Input sequence length {T} exceeds the maximum context size {self.n_ctx}")
        
        # position offset: 0 during prefill or no-cache, kv_cache.seq_len during decode
        start_pos = kv_cache.seq_len if kv_cache is not None else 0
        
        # token embeddings: (B, T) -> (B, T, n_embd)
        tok_emb = self.wte(input_ids)
        
        # # This approach does not work with havving left padding tokens, 
        # # since the position ids will be shifted by the number of padding tokens, 
        # # which is not correct.
        # # position embeddings with offset: arange(start_pos, start_pos + T) -> (1, T) -> (B, T, n_embd)
        # pos_ids = torch.arange(start_pos, start_pos + T, device = input_ids.device, dtype = torch.long).unsqueeze(0)
        # pos_emb = self.wpe(pos_ids)
        
        if padding_mask is not None:
            # cumsum of the mask gives position 1,2,3... for real tokens, stays flat for pads
            # subtract 1 so real tokens start at position 0
            # add start_pos offset for KV cache decode steps
            pos_ids = (padding_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            if kv_cache is not None and T == 1:
                # During decode: padding_mask is the full mask including generated tokens
                # The new token's position = number of real tokens before it
                pos_ids = pos_ids[:, -1:] # just the last column, shape (B, 1)
        else:
            pos_ids = torch.arange(start_pos, start_pos + T, device=input_ids.device).unsqueeze(0)
        pos_emb = self.wpe(pos_ids)
        
        # combine token and position embeddings, then apply dropout: (B, T, n_embd)
        x = tok_emb + pos_emb
        x = self.embd_drop(x)
        
        # pass through transformer blocks
        for idx, block in enumerate(self.h):
            x = block(x, idx, kv_cache, padding_mask = padding_mask) if kv_cache is not None else block(x, padding_mask = padding_mask)
            
        # final layer normalization
        x = self.ln_f(x)
        # project to vocabulary size to get logits: (B, T, n_embd) -> (B, T, vocab_size)
        x = x @ self.wte.weight.T
        
        return x
