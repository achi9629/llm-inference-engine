
"""
Rotary Position Embeddings (RoPE) for Mistral.

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
                 rope_theta: float = 10000.0
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
            rope_theta: base frequency for the geometric progression of rotation
                frequencies. Default 10000.0 (Mistral 1/2). Higher values
                (e.g. 500000 in Mistral) extend long-context performance.
        Returns:
            None
        '''
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        
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
        
        inv_angle = 1.0 / self.rope_theta ** (torch.arange(0, self.dim, 2) / self.dim)
        seq = torch.arange(self.max_seq_len)
        freq = torch.einsum('i,j->ij', seq, inv_angle)
        freqs_cis = torch.polar(torch.ones_like(freq), freq)
        
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
    
    def apply_rotary_pos_emb(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        
        '''
        Description:
            Apply rotary position embeddings to an input tensor (Q or K).
            Steps:
                1. Reshape x into consecutive dimension pairs: (B, H, T, dim/2, 2)
                2. View pairs as complex numbers: (B, H, T, dim/2)
                3. Multiply by precomputed freqs_cis — this IS the rotation
                4. Convert back to real and reshape to (B, H, T, dim)
        Args:
            x: input tensor of shape (B, H, T, dim) where
                B = batch size, H = number of heads,
                T = sequence length, dim = head dimension.
            start_pos: starting position index for the sequence. This allows for
                applying RoPE to sequences that are continuations of previous ones.
        Returns:
            Rotated tensor of shape (B, H, T, dim) with same dtype as input.
        '''
        
        B, H, T, dim = x.shape
        
        assert dim == self.dim, f"Input dimension {dim} does not match RoPE dimension {self.dim}"
        assert dim % 2 == 0, "RoPE dimension must be even"
        assert start_pos >= 0, "start_pos must be non-negative"
        
        # Absolute positions: start_pos to start_pos + T - 1
        end_pos = start_pos + T
        assert end_pos <= self.max_seq_len, f"Sequence end position {end_pos} exceeds max_seq_len {self.max_seq_len}"
        
        freqs_cis = self.freqs_cis[start_pos: end_pos, :]
        
        # torch.view_as_complex requires the last dimension to be even, so we reshape accordingly
        # It also required the dtype to be float, so we convert to float before reshaping and convert back to the original dtype after the operation
        x_reshaped = x.float().view(B, H, T, dim//2, 2)
        x_complex = torch.view_as_complex(x_reshaped)
        x_rotated = torch.einsum('bhtd, td -> bhtd', x_complex, freqs_cis)
        
        return torch.view_as_real(x_rotated).view(B, H, T, dim).to(x.dtype)