import torch # type: ignore
import torch.nn as nn # type: ignore

from .generator import generator
from llm_engine.cache.kv_cache import KVCache

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
        ) -> None:
        
        '''
        Description:
            Initialize the inference engine with model, tokenizer, and optional KV cache.
            
            When is_kv_cache_enabled=True, a KVCache is pre-allocated using model_cfg
            dimensions. The cache is reset at the start of each generate() call.

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
                Must contain: n_layer, n_head, n_embd.
        '''
        
        # Initialize the inference engine with the provided 
        # model, device and setting model to evaluation mode
        self.model = model
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize the tokenizer, 
        # set the maximum number of tokens, 
        # end-of-sequence token ID, 
        # sampling method for text generation
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.sampling_method = sampling_method
        
        # Validate model
        if not isinstance(self.model, nn.Module):
            raise TypeError("Model must be an instance of the model class.")
        
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
        # Cache is pre-allocated on the model's device with matching dtype.
        if is_kv_cache_enabled:
            
            if model_cfg is None:
                raise ValueError("model_cfg is required when is_kv_cache_enabled=True")            
            self.kv_cache = KVCache(batch_size = batch_size,
                                    n_layers = model_cfg["n_layer"],
                                    n_heads = model_cfg["n_head"],
                                    head_dim = model_cfg["n_embd"] // model_cfg["n_head"],
                                    max_seq_len = max_tokens_for_kv_cache,
                                    dtype = next(self.model.parameters()).dtype,
                                    device = self.device
                )
        else:
            self.kv_cache = None
        
    def encode(self, 
               input_text: str | list[str],
        ) -> torch.Tensor:
        
        '''
        Encode the input text into token IDs using the tokenizer.
        Args:
            input_text (str | list[str]): The input text to be encoded.
        Returns:
            torch.Tensor: Encoded token IDs as a PyTorch tensor (return_tensor=True).
        '''
        
        return self.tokenizer.encode(input_text, 
                                     return_tensor = True
                                    )
    
    def decode(self,
               token_ids: torch.Tensor
        ) -> str | list[str]:
        
        '''
        Decode token IDs into text as str (single sequence) or list[str] (batch).
        Args:
            token_ids (torch.Tensor): The token IDs to be decoded.
        Returns:
            str | list[str]: The decoded text as a string or list of strings.
        '''
        
        return self.tokenizer.decode(token_ids)
    
    def generate(self,
                 input_text: str | list[str],
                 max_tokens: int = 50, 
        ) -> dict[str, object] | list[dict[str, object]]:
        
        '''
        Description:
            Generate text from one or more prompts using the configured model and decoding settings.
            Resets the KV cache (if enabled) before each generation call.

        Args:
            input_text (str | list[str]): The input prompt(s) for text generation.
            max_tokens (int): Maximum number of new tokens to generate. Default: 50.

        Returns:
            dict[str, object] | list[dict[str, object]]: 
            A response dictionary for single input, or a list of response dictionaries for batch input.
            Each response includes input_text, generated_text, sampling_method, token_count, and stop_reason.
        '''
                
        if self.kv_cache is not None:
            self.kv_cache.reset_cache()
        
        if input_text == "":
            raise ValueError("Input prompt cannot be empty.")
        elif isinstance(input_text, str) and input_text.strip() == "":
            raise ValueError("Input prompt cannot be empty or whitespace-only.")
        elif isinstance(input_text, list) and len(input_text) == 0:
            raise ValueError("Input prompt list cannot be empty.")
        elif isinstance(input_text, list) and not all(isinstance(item, str) for item in input_text):
            raise ValueError("All items in the input prompt list must be strings.")
        
        token_ids = self.encode(input_text)
        
        predicted_token_ids, token_count, stop_reason = generator(
                                                                    model = self.model,
                                                                    token_ids = token_ids,
                                                                    device = self.device,
                                                                    max_tokens = max_tokens,
                                                                    eos_token_id = self.eos_token_id,
                                                                    sampling_method = self.sampling_method,
                                                                    kv_cache = self.kv_cache
                                                                )
        
        decoded_text = self.decode(predicted_token_ids)
        # TODO: Normalize single-input decode output.
        # If input_text is a single string but decoded_text is ['...'] (one-item list),
        # convert it to a plain string before building the response.
        # Example:
        # if isinstance(input_text, str) and isinstance(decoded_text, list) and len(decoded_text) == 1:
        #     decoded_text = decoded_text[0]
        
        if isinstance(input_text, str):
            return {
                "input_text": input_text,
                "generated_text": decoded_text,
                "sampling_method": self.sampling_method,
                "token_count": token_count,
                "stop_reason": stop_reason
            }
        elif isinstance(input_text, list) and isinstance(decoded_text, list) and len(input_text) == len(decoded_text):
            return [
                    {
                        "input_text": input_text[i],
                        "generated_text": decoded_text[i],
                        "sampling_method": self.sampling_method, # This is currently the same for all inputs in the batch, as the generator function is not yet implemented to handle batch generation.
                        "token_count": token_count, # This is currently the same for all inputs in the batch, as the generator function is not yet implemented to handle batch generation.
                        "stop_reason": stop_reason # This is currently the same for all inputs in the batch, as the generator function is not yet implemented to handle batch generation.
                    }
                    for i in range(len(input_text))
                ]
        else:
            raise ValueError("Input text and decoded text must both be strings or lists of the same length.")
