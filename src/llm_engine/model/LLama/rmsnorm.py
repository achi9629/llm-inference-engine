import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps : float = 1e-6) -> None:
        super(RMSNorm, self).__init__()
        
        '''
        Description:
            Root Mean Square Layer Normalization (RMSNorm).
            Unlike LayerNorm, RMSNorm skips mean subtraction and bias — it only
            re-scales by the root-mean-square of the input. One learnable parameter (weight).
            Computes in float32 for numerical stability, then casts back to input dtype.
        Args:
            d_model (int): Hidden dimension size.
            eps (float): Small constant for numerical stability (default: 1e-6).
        Returns:
            None
        '''
        
        
        self.d_model = d_model
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        '''
        Description:
            Applies RMSNorm: x_normed = (x / RMS(x)) * weight
            where RMS(x) = sqrt(mean(x²) + eps).
            Upcasts to float32 for the RMS computation to avoid fp16 overflow.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Normalized tensor of same shape and dtype as input.
        '''
        
        # x: [batch_size, seq_len, d_model]
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim = True) + self.eps)
        x_normed = x / rms.to(x.dtype)
        
        return x_normed * self.weight