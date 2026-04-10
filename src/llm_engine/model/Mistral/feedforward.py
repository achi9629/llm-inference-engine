
"""
Mistral Feed-Forward Network (SwiGLU).

Implements the gated feed-forward sublayer used in each Mistral transformer block.
Instead of the standard two-linear-layer FFN, Mistral uses SwiGLU — a gated linear
unit where the gate is activated with SiLU (Swish). This gives three projections:

    gate_proj : x → W_gate · x          (gating path, activated with SiLU)
    up_proj   : x → W_up   · x          (value path)
    down_proj : h → W_down · h          (project back to d_model)

    output = W_down · (SiLU(W_gate · x) ⊙ W_up · x)

Reference: Shazeer (2020) "GLU Variants Improve Transformer"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MistralFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str = 'silu') -> None:
        super(MistralFeedForward, self).__init__()
        
        """
        Description:
            Initialize the Mistral SwiGLU feed-forward network.
        Args:
            d_model: Model embedding dimension (input/output size).
            d_ff: Intermediate (expanded) dimension. Must be >= d_model.
            activation: Activation function name. Supported: 'silu'.
        Returns:
            None
        Raises:
            AssertionError: If d_model > d_ff.
            ValueError: If activation is not supported.
        """
        
        assert d_model <= d_ff, "d_model should be less than or equal to d_ff for efficient computation."
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        
        if self.activation == 'silu':
            self.act_fn = F.silu
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Description:
            Compute the SwiGLU feed-forward transformation.
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        
        hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(hidden)
        
        return output