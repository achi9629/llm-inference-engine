
'''
Inference Engine

Orchestrator for text generation: tokenize → generate → detokenize.
Supports three cache modes:
    - No cache (is_kv_cache_enabled=False): full sequence recomputed each step
    - Standard KV cache (cache_type="standard"): pre-allocated contiguous tensors per layer
    - Paged KV cache (cache_type="paged"): block-level memory management via
      MemoryAllocator + BlockTable + PagedKVCache, wrapped in PagedCacheContext adapter

For paged mode, generate() manages the full block lifecycle per call:
    create sequences → allocate blocks → build PagedCacheContext → run generator →
    reset_blocks → free_sequence
'''

import torch
import time
import secrets
import math
import torch.nn as nn

from .generator import generator
from ..cache.kv_cache import KVCache
from ..cache.block_table import BlockTable
from ..cache.paged_kv_cache import PagedKVCache
from ..cache.memory_allocator import MemoryAllocator
from ..cache.paged_cache_context import PagedCacheContext

class InferenceEngine:
    def __init__(self, 
                 model: nn.Module, 
                 device: str, 
                 tokenizer: object, 
                 eos_token_id: int = 50256, 
                 sampling_method: str = "greedy",
                 is_kv_cache_enabled: bool = False,
                 max_tokens_for_kv_cache: int = 2048,
                 batch_size: int = 1,
                 model_cfg: dict = None,
                 cache_type: str = "standard",
                 num_blocks: int = None,
                 block_size: int = None,
        ) -> None:
        
        '''
        Description:
            Initialize the inference engine with model, tokenizer, and optional KV cache.

            Cache modes:
            - is_kv_cache_enabled=False: no caching, self.kv_cache = None.
            - cache_type="standard": pre-allocates a single KVCache object using model_cfg
            dimensions. Reset at the start of each generate() call.
            - cache_type="paged": creates three persistent components:
            MemoryAllocator (block pool), BlockTable (logical→physical mapping),
            PagedKVCache (GPU tensor storage). The PagedCacheContext adapter is
            built fresh in each generate() call with per-request sequence metadata.

        Args:
            model (nn.Module): The transformer model for text generation.
            device (str): Target device (e.g., 'cpu', 'cuda').
            tokenizer (object): Tokenizer with encode() and decode() methods.
            eos_token_id (int): End-of-sequence token ID. Default: 50256 (GPT-2).
            sampling_method (str): Decoding strategy. Default: 'greedy'.
            is_kv_cache_enabled (bool): Whether to use KV caching for faster decoding.
            max_tokens_for_kv_cache (int): Maximum sequence length the cache can hold.
            batch_size (int): Batch size for KV cache allocation. Default: 1.
            model_cfg (dict): Model config dict (required when is_kv_cache_enabled=True).
                Must contain: n_layer, n_heads, n_embd.
            cache_type (str): 'standard' for contiguous KVCache, 'paged' for block-level paged cache.
            num_blocks (int): Total number of blocks in the paged cache pool (required when cache_type='paged').
            block_size (int): Number of token slots per block (required when cache_type='paged').
        Returns:
            None
        '''
        
        # Initialize the inference engine with the provided 
        # model, device and setting model to evaluation mode
        self.model = model
        self.device = device
        
        # Initialize the tokenizer, 
        # set the maximum number of tokens, 
        # end-of-sequence token ID, 
        # sampling method for text generation
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.sampling_method = sampling_method
        self.is_kv_cache_enabled = is_kv_cache_enabled
        self.cache_type = cache_type
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        # Validate model
        if not isinstance(self.model, nn.Module):
            raise TypeError("Model must be an instance of the model class.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Validate that the tokenizer has the required methods
        if self.tokenizer is None:
            raise TypeError("tokenizer cannot be None")
        
        encode_fn = getattr(self.tokenizer, "encode", None)
        decode_fn = getattr(self.tokenizer, "decode", None)
        
        if not callable(encode_fn):
            raise TypeError("Tokenizer must provide a callable 'encode' method.")
        
        if not callable(decode_fn):
            raise TypeError("Tokenizer must provide a callable 'decode' method.")
        
        # Initialize KV cache if enabled; requires model_cfg for layer/head dimensions.
        if self.is_kv_cache_enabled:
            
            if model_cfg is None:
                raise ValueError("model_cfg is required when is_kv_cache_enabled=True")

            # Cache is pre-allocated on the model's device with matching dtype.
            if self.cache_type == "standard":       
                self.kv_cache = KVCache(batch_size = batch_size,
                                        n_layers = model_cfg["n_layer"],
                                        n_heads = model_cfg["n_heads"],
                                        head_dim = model_cfg["n_embd"] // model_cfg["n_heads"],
                                        max_seq_len = max_tokens_for_kv_cache,
                                        dtype = next(self.model.parameters()).dtype,
                                        device = self.device
                    )
                
            # For paged cache, we initialize the MemoryAllocator and BlockTable to manage the block pool,
            # and the PagedKVCache to store the actual key/value tensors. The PagedCacheContext adapter will 
            # be created/reset in generate() with the appropriate sequence IDs and lengths for each generation
            elif cache_type == "paged":
                self.allocator = MemoryAllocator(num_blocks = num_blocks)
                self.block_table = BlockTable(allocator = self.allocator,
                                              block_size = self.block_size
                                    )
                self.paged_kv_cache = PagedKVCache(num_blocks = num_blocks,
                                              n_layers = model_cfg["n_layer"],
                                              n_heads = model_cfg["n_heads"],
                                              block_size = self.block_size,
                                              head_dim = model_cfg["n_embd"] // model_cfg["n_heads"],
                                              dtype = next(self.model.parameters()).dtype,
                                              device = self.device
                                    )
                self.kv_cache = None # The PagedCacheContext will be created/reset in generate() with the appropriate sequence IDs and lengths.
            else:
                raise ValueError(f"Unsupported cache_type '{cache_type}'. Supported types: 'standard', 'paged'.")
        else:
            self.kv_cache = None
        
    def encode(self, 
               input_text: str | list[str],
        ) -> torch.Tensor:
        
        '''
        Description:
            Encode the input text into token IDs using the tokenizer.
        Args:
            input_text (str | list[str]): The input text to be encoded.
        Returns:
            torch.Tensor: Encoded token IDs as a PyTorch tensor (return_tensor=True).
        '''
        
        return self.tokenizer.encode(input_text, 
                                     return_tensor = True)
    
    def decode(self,
               token_ids: torch.Tensor
        ) -> str | list[str]:
        
        '''
        Description:
            Decode token IDs into text as str (single sequence) or list[str] (batch).
        Args:
            token_ids (torch.Tensor): The token IDs to be decoded.
        Returns:
            str | list[str]: The decoded text as a string or list of strings.
        '''
        
        return self.tokenizer.decode(token_ids)
    
    def generate(self,
                 input_text: str | list[str],
                 max_tokens: int | list[int] = 50 , 
        ) -> dict[str, object] | list[dict[str, object]]:
        
        '''
        Description:
            Generate text from one or more prompts using the configured model and decoding settings.

            Cache behavior per call:
            - standard: resets the pre-allocated KVCache (zeros all tensors).
            - paged: creates unique sequence IDs, registers them in the block table,
            allocates initial blocks (ceil(prompt_tokens / block_size)), builds a
            fresh PagedCacheContext adapter, and passes it as kv_cache to the generator.
            After generation completes: reset_blocks (zero GPU tensors) then
            free_sequence (return blocks to allocator's free list).

        Args:
            input_text (str | list[str]): The input prompt(s) for text generation.
            max_tokens (int | list[int]): The maximum number of tokens to generate for each input. Can be a single int or a list of ints matching the batch size.

        Returns:
            dict[str, object] | list[dict[str, object]]:
            A response dictionary for single input, or a list of response dictionaries for batch input.
            Each response includes input_text, generated_text, sampling_method, token_count, and stop_reason.
        '''
        
        if input_text == "":
            raise ValueError("Input prompt cannot be empty.")
        elif isinstance(input_text, str) and input_text.strip() == "":
            raise ValueError("Input prompt cannot be empty or whitespace-only.")
        elif isinstance(input_text, list) and len(input_text) == 0:
            raise ValueError("Input prompt list cannot be empty.")
        elif isinstance(input_text, list) and not all(isinstance(item, str) for item in input_text):
            raise ValueError("All items in the input prompt list must be strings.")
        
        token_ids, padding_mask = self.encode(input_text)
        
        if self.is_kv_cache_enabled:
            if self.cache_type == "standard":
                self.kv_cache.reset_cache()
                
            elif self.cache_type == "paged":
                seq_ids = [f"{int(time.time()*1000)}-{secrets.token_hex(4)}" for i in range(len(token_ids))] # Generate unique sequence IDs for each input in the batch.
                seq_lens = [0] * len(token_ids)
                for idx in range(len(seq_ids)):
                    self.block_table.add_sequence(seq_id = seq_ids[idx])
                    num_blocks = math.ceil(len(token_ids[idx]) / self.block_size)
                    self.block_table.allocate_blocks(seq_id = seq_ids[idx],
                                                        num_blocks = num_blocks)
                
                self.kv_cache = PagedCacheContext(paged_kv_cache = self.paged_kv_cache,
                                                  block_table = self.block_table,
                                                  seq_ids = seq_ids,
                                                  seq_lens = seq_lens,
                                                  block_size = self.block_size
                                    )
            else:
                raise ValueError(f"Unsupported cache_type '{self.cache_type}' during cache reset. Supported types: 'standard', 'paged'.")
        
        
        predicted_token_ids, token_count, stop_reason = generator(
                                                                    model = self.model,
                                                                    token_ids = token_ids,
                                                                    device = self.device,
                                                                    max_tokens = max_tokens,
                                                                    eos_token_id = self.eos_token_id,
                                                                    padding_mask = padding_mask, 
                                                                    sampling_method = self.sampling_method,
                                                                    kv_cache = self.kv_cache
                                                                )
        
        if self.is_kv_cache_enabled and self.cache_type == "paged":
            
            self.kv_cache.reset_blocks() # Reset the PagedCacheContext's block tracking after freeing blocks in the block table.
            
            # Clean up allocated blocks for the current generation after generation is complete.
            for seq_id in seq_ids:
                self.block_table.free_sequence(seq_id = seq_id)
            
        decoded_text = self.decode(predicted_token_ids)

        if isinstance(input_text, str) and isinstance(decoded_text, list) and len(decoded_text) == 1:
            decoded_text = decoded_text[0]
        
        if isinstance(input_text, str):
            return {
                "input_text": input_text,
                "generated_text": decoded_text,
                "sampling_method": self.sampling_method,
                "token_count": token_count.item() if isinstance(token_count, torch.Tensor) else token_count, # Convert single-token count tensor to int.
                "stop_reason": stop_reason[0] if isinstance(stop_reason, list) else stop_reason # Assuming stop_reason is a list with one reason for single input.
            }
        elif isinstance(input_text, list) and isinstance(decoded_text, list) and len(input_text) == len(decoded_text):
            return [
                    {
                        "input_text": input_text[i],
                        "generated_text": decoded_text[i],
                        "sampling_method": self.sampling_method, # This is currently the same for all inputs in the batch, as the generator function is not yet implemented to handle batch generation.
                        "token_count": token_count[i].item(), # Convert each token count tensor to int for the response.
                        "stop_reason": stop_reason[i] # Assuming stop_reason is a list of reasons corresponding to each input in the batch.
                    }
                    for i in range(len(input_text))
                ]
        else:
            raise ValueError("Input text and decoded text must both be strings or lists of the same length.")
