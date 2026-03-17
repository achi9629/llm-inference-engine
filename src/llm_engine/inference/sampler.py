import torch # type: ignore

def greedy_sampler(logits : torch.Tensor) -> torch.Tensor:
    
    """
    Select next token id(s) using greedy decoding.

    Greedy decoding chooses the index with the maximum logit over the
    vocabulary dimension (`argmax`) for each input sequence.

    Supported input shapes:
    - 1D: [vocab_size]
    - 2D: [batch_size, vocab_size]

    For 1D input, logits are temporarily expanded to batch form before
    applying argmax, and the result is squeezed back to a scalar tensor.

    Args:
        logits: Model output logits for the current decoding step.

    Returns:
        Predicted token id(s):
        - Scalar tensor for single-input logits.
        - 1D tensor of shape [batch_size] for batched logits.

    Raises:
        TypeError: If logits is not a torch.Tensor.
        ValueError: If logits is not 1D or 2D.
    """
    
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Input logits must be a torch.Tensor")
    
    if logits.dim() not in (1, 2):
        raise ValueError("Input logits must be a 1D or 2D tensor")
    
    if logits.dim() == 1: 
        logits = logits.unsqueeze(0)
        
    # Get the index of the maximum logit for each batch
    # input shape: [batch_size, vocab_size]
    # output shape: [batch_size]
    max_indices = torch.argmax(logits, dim=-1)

    return max_indices
    
    
    