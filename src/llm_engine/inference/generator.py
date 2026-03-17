import torch # type: ignore

from .sampler import greedy_sampler

def generator(model: object, 
              token_ids: torch.Tensor, 
              device: str,
              max_tokens: int = 50, 
              eos_token_id: int = 50256,
              sampling_method: str = 'greedy',
              kv_cache: object = None,
    ) -> tuple[torch.Tensor, int, str]:
    
    """
    Generate tokens autoregressively using greedy sampling.

    This function accepts prompt token IDs as either:
    - 1D tensor: [seq_len]
    - 2D tensor: [batch_size, seq_len]

    The generation loop:
    1. Runs model forward pass on current token_ids.
    2. Extracts last-step logits: [batch_size, vocab_size].
    3. Selects next token IDs using sampler method (eg. greedy argmax).
    4. Appends sampled tokens to token_ids.
    5. Stops when all sampled tokens are EOS or max_tokens is reached.

    Notes:
    - 1D inputs are temporarily converted to batch form [1, seq_len].
    - Return shape matches input rank:
      - 1D input -> 1D output
      - 2D input -> 2D output

    Args:
        model: Callable model that returns logits with shape
            [batch_size, sequence_length, vocab_size].
        token_ids (torch.Tensor): Input token IDs.
        device (str): Target device (e.g., "cpu", "cuda", "cuda:0").
        max_tokens (int): Maximum number of new tokens to generate.
        eos_token_id (int): End-of-sequence token ID.
        sampling_method (str): Sampling strategy. Currently supports "greedy".

    Returns:
        torch.Tensor: Generated token IDs including original prompt tokens.

    Raises:
        TypeError: If token_ids is not a torch.Tensor.
        ValueError: If token_ids shape is invalid, empty, max_tokens is invalid,
            sampling_method is unsupported, or model logits are not 3D.
    """
    
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a torch.Tensor")
    
    if token_ids.ndim not in (1, 2):
        raise ValueError("token_ids must be a 1D or 2D tensor")
    
    if token_ids.size(-1) == 0:
        raise ValueError("token_ids cannot be empty")
    
    if token_ids.ndim == 2 and token_ids.size(0) == 0:
        raise ValueError("token_ids cannot have zero batch size")
    
    if max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    
    if sampling_method not in ('greedy',):
        raise ValueError(f"Unsupported sampling method: {sampling_method}")
    
    token_count = 0
    
    with torch.inference_mode():
        token_ids = token_ids.to(device)
        
        was_1d = token_ids.ndim == 1
        if was_1d:
            token_ids = token_ids.unsqueeze(0)
            
        input_ids = token_ids # Keep original input for final output concatenation.
        while True:
            
            logits = model(input_ids, kv_cache = kv_cache)
            
            if logits.ndim != 3:
                raise ValueError("Model output logits must be a 3D tensor of shape (batch_size, sequence_length, vocab_size)")
        
            # Why logits[:, -1, :] ?
            # The model returns logits at every position: (B, T, vocab_size).
            # Due to causal attention, each position's hidden state only sees tokens
            # at or before itself. So position T-1's hidden state has attended to
            # ALL tokens [0, 1, ..., T-1] across all 12 layers — it encodes the
            # full sequence context. Its logits are the prediction for what comes
            # AFTER the entire input. All earlier positions' logits predict tokens
            # we already have, so we discard them and keep only the last.
            # With KV cache during decode, T=1, so this trivially picks the only position.
            logits = logits[:, -1, :]
        
            if sampling_method == 'greedy':
                output_token_id = greedy_sampler(logits)

            token_ids = torch.cat([token_ids, output_token_id.unsqueeze(-1)], dim=-1)

            if kv_cache is not None:
                # During decoding with KV cache, we only feed the new token.
                input_ids = output_token_id.unsqueeze(-1) # Shape: (batch_size, 1)
            else:
                # During non-cached generation, we feed the entire sequence each step.
                input_ids = token_ids
            

            # TODO: Replace all-EOS batch stop with per-sequence finished-mask handling.
            # * This allows each sequence to stop independently during batched decoding.
            if (output_token_id == eos_token_id).all():
                stop_reason = "All sequences generated EOS token."
                break
            
            token_count += 1 # 
            
            if token_count >= max_tokens:
                stop_reason = f"Reached max_tokens limit of {max_tokens}."
                break
    
    if was_1d:
        token_ids = token_ids.squeeze(0)
    
    return token_ids, token_count, stop_reason
    
    
    