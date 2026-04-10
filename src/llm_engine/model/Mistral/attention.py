import torch, logging
import torch.nn as nn

from .rope import RoPE

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MistralAttention(nn.Module):
    def __init__(self,
                 n_embd: int,
                 n_heads: int,
                 n_kv_heads: int,
                 n_ctx: int,
                 rope_theta: int = 10000.0,
                 attn_pdrop: float = 0.0
        ) -> None:
        super(MistralAttention, self).__init__()
        
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for making groups"
        
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_ctx = n_ctx
        self.head_dim = n_embd // n_heads
        self.n_groups = n_heads // n_kv_heads
        self.attn_pdrop = attn_pdrop
        self.rope_theta = rope_theta
        
        self.k_proj = nn.Linear(self.n_embd, self.n_embd // self.n_groups, bias = False)
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias = False)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd // self.n_groups, bias = False)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias = False)
        
        self.scale = self.head_dim ** -0.5
        
        self.attn_drop = nn.Dropout(self.attn_pdrop)
        
        self.rope = RoPE(dim = self.head_dim, 
                         max_seq_len = self.n_ctx, 
                         rope_theta = self.rope_theta)
        
    def repeat_kv(self, 
                  x: torch.Tensor, 
                  n_groups: int
        ) -> torch.Tensor:
        
        if n_groups == 1:
            # No need to repeat if n_groups is 1
            return x
        
        # x = (B, n_kv_heads, T, head_dim)
        B, n_kv_heads, T, head_dim = x.size()
        
        x = x.unsqueeze(2) # (B, n_kv_heads, 1, T, head_dim)
        x = x.expand(B, n_kv_heads, n_groups, T, head_dim) # (B, n_kv_heads, 1, T, head_dim) -> (B, n_kv_heads, n_groups, T, head_dim)
        
        return x.reshape(B, n_kv_heads * n_groups, T, head_dim) # (B, n_kv_heads, n_groups, T, head_dim) -> (B, n_heads, T, head_dim)
        
    def forward(self ,
                x: torch.Tensor,
                layer_idx: int = None,
                kv_cache: object = None,
                padding_mask: torch.Tensor = None,) -> torch.Tensor:
        
        B, T, dim = x.size()
        device = x.device
        
        # logger.info(f"self.k_proj weight shape: {self.k_proj.weight.shape}, {x.shape}")
        
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim) # (B, T, head_dim * n_kv_heads) -> (B, T, n_kv_heads, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)    # (B, T, head_dim * n_heads) -> (B, T, n_heads, head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim) # (B, T, head_dim * n_kv_heads) -> (B, T, n_kv_heads, head_dim)
        
        k = k.permute(0, 2, 1, 3) # (B, T, n_kv_heads, head_dim) -> (B, n_kv_heads, T, head_dim)
        q = q.permute(0, 2, 1, 3) # (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        v = v.permute(0, 2, 1, 3) # (B, T, n_kv_heads, head_dim) -> (B, n_kv_heads, T, head_dim)
        
        if kv_cache is not None and layer_idx is not None:
            # Update the key and value tensors with the cached values from previous time steps.
            # Return the updated key and value tensors for use in the attention computation.
            # k = (B, n_kv_heads, T, head_dim) -> (B, n_kv_heads, T_total, head_dim)
            # v =  (B, n_kv_heads, T, head_dim) -> (B, n_kv_heads, T_total, head_dim)
            k, v = kv_cache.update_cache(layer_idx, k, v)
            T_total = k.size(2)
        else:
            T_total = T
        start_pos = T_total - T
        end_pos = T_total
        
        # Apply RoPE to q and k
        q = self.rope.apply_rotary_pos_emb(q, start_pos = start_pos) # (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        # for k start_pos is always 0 because we want to apply the same positional embedding to the cached keys and the current keys, 
        # as they are all part of the same sequence when we consider the cached keys and the current keys together.
        k = self.rope.apply_rotary_pos_emb(k, start_pos = 0) # (B, n_kv_heads, T_total, head_dim) -> (B, n_kv_heads, T_total, head_dim)
        
        # Repeat the key and value tensors to match the number of attention heads. 
        # This is necessary because the number of key-value heads (n_kv_heads) may be less than the total number of attention heads (n_heads), 
        # and we need to ensure that each attention head has a corresponding key and value tensor for the attention computation.
        k = self.repeat_kv(k, self.n_groups) # (B, n_kv_heads, T_total, head_dim) -> (B, n_heads, T_total, head_dim)
        v = self.repeat_kv(v, self.n_groups) # (B, n_kv_heads, T_total, head_dim) -> (B, n_heads, T_total, head_dim)
        
        score = q @ k.transpose(-2, -1) * self.scale # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T_total) -> (B, n_heads, T, T_total)
        
        # Compute attention scores
        # (T_total, T_total) -> (T_total - T, T_total) -> (1, 1, T_total - T, T_total)
        causal_mask = torch.tril(torch.ones(T_total, T_total, device = device))[start_pos: end_pos, : end_pos].unsqueeze(0).unsqueeze(0)
        
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
            
        attn_weights = torch.softmax(score, dim = -1) # (B, n_heads, T, T_total)
        attn_weights = self.attn_drop(attn_weights)
        
        out = attn_weights @ v # (B, n_heads, T, T_total) @ (B, n_heads, T_total, head_dim) -> (B, n_heads, T, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, dim) # (B, n_heads, T, head_dim) -> (B, T, n_heads * head_dim) -> (B, T, dim)
        
        out = self.o_proj(out) # (B, T, dim) -> (B, T, dim)
        
        return out