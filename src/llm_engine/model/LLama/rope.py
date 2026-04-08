
"""
Rotary Position Embeddings (RoPE) for LLaMA.

Encodes position information directly into Q and K vectors before the attention
dot product. Uses 2D rotation at position-dependent angles so that the Q·K dot
product depends only on relative distance (m - n), not absolute positions.

The implementation uses the complex-number trick: each consecutive dimension pair
is viewed as a complex number, multiplied by e^{i·m·θ}, and converted back to real.
This is mathematically equivalent to applying a 2D rotation matrix per pair.
"""

import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, 
                 dim: int, 
                 max_seq_len: int, 
                 theta: float = 10000.0
        ) -> None:
        super(RoPE, self).__init__()
        
        '''
        Description:
            Initialize the RoPE module by precomputing the complex frequency tensor.
        Args:
            dim: head dimension (must be even). Each consecutive pair of dimensions
                forms a 2D rotation plane, yielding dim/2 independent rotations.
            max_seq_len: maximum sequence length supported. Frequencies are
                        precomputed for positions 0 to max_seq_len - 1.
            theta: base frequency for the geometric progression of rotation
                frequencies. Default 10000.0 (LLaMA 1/2). Higher values
                (e.g. 500000 in LLaMA 3) extend long-context performance.
        Returns:
            None
        '''
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        self.precompute_freqs_cis()
        
    def precompute_freqs_cis(self) -> None:
        
        '''
        Description:
            Precompute the complex frequency tensor freqs_cis of shape
            (max_seq_len, dim/2) and register it as a non-persistent buffer.

            For each dimension pair i and position m:
                freqs_cis[m, i] = e^{i · m · θ_i} = cos(m·θ_i) + i·sin(m·θ_i)
            where θ_i = 1 / theta^(2i/dim).

            Pair 0 has the highest frequency (captures local position).
            Pair dim/2 - 1 has the lowest frequency (captures long-range position).

            Registered as non-persistent so it is not saved in state_dict but
            still moves with the module on .to(device) calls.
        Args:
            None
        Returns:
            None (sets self.freqs_cis as a registered buffer)
        '''
        
        inv_angle = 1.0 / self.theta ** (torch.arange(0, self.dim, 2) / self.dim)
        seq = torch.arange(self.max_seq_len)
        freq = torch.einsum('i,j->ij', seq, inv_angle)
        freqs_cis = torch.polar(torch.ones_like(freq), freq)
        
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
    
    def apply_rotary_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        
        '''
        Description:
            Apply rotary position embeddings to an input tensor (Q or K).
            Steps:
                1. Reshape x into consecutive dimension pairs: (B, T, H, dim/2, 2)
                2. View pairs as complex numbers: (B, T, H, dim/2)
                3. Multiply by precomputed freqs_cis — this IS the rotation
                4. Convert back to real and reshape to (B, T, H, dim)
        Args:
            x: input tensor of shape (B, T, H, dim) where
            B = batch size, T = sequence length,
            H = number of heads, dim = head dimension.
        Returns:
            Rotated tensor of shape (B, T, H, dim) with same dtype as input.
        '''
        
        B, T, H, dim = x.shape
        
        assert dim == self.dim, f"Input dimension {dim} does not match RoPE dimension {self.dim}"
        assert dim % 2 == 0, "RoPE dimension must be even"
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum sequence length {self.max_seq_len}"
        
        freqs_cis = self.freqs_cis[:T, :]
        
        # torch.view_as_complex requires the last dimension to be even, so we reshape accordingly
        # It also required the dtype to be float, so we convert to float before reshaping and convert back to the original dtype after the operation
        x_reshaped = x.float().view(B, T, H, dim//2, 2)
        x_complex = torch.view_as_complex(x_reshaped)
        x_rotated = torch.einsum('bthd, td -> bthd', x_complex, freqs_cis)
        
        return torch.view_as_real(x_rotated).view(B, T, H, dim).to(x.dtype)