
"""
Request handler module for the LLM inference engine serving layer.

Parses and validates incoming HTTP request parameters, tokenizes the prompt,
and creates a scheduler Request object. Acts as the boundary between the
HTTP layer and the internal scheduling system.
"""

import uuid

from ..tokenizer.tokenizer import Tokenizer
from ..scheduler.request import Request

class RequestHandler:
    def __init__(self, 
                 tokenizer: Tokenizer, 
                 max_model_len: int
        ) -> None:
        
        '''
        Description:
            Initializes the RequestHandler with a tokenizer and maximum model context length.
            Validates that the tokenizer is a proper Tokenizer instance and max_model_len is
            a positive integer.
        Args:
            tokenizer (Tokenizer): The tokenizer instance used to encode prompts into token IDs.
            max_model_len (int): Maximum context length of the model (e.g., 1024 for GPT-2). 
                Used to cap max_tokens in incoming requests.
        Returns:
            None
        '''
        
        if not isinstance(tokenizer, Tokenizer):
            raise ValueError("tokenizer must be an instance of Tokenizer")
        if not isinstance(max_model_len, int) or max_model_len <= 0:
            raise ValueError("max_model_len must be a positive integer")
        
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len
        
    def handle(self, 
               prompt: str, 
               max_tokens: int
        ) -> Request:
        
        '''
        Description:
            Validates the incoming prompt and max_tokens, tokenizes the prompt, generates a 
            unique request ID, and creates a Request object for the scheduler.
        Args:
            prompt (str): The input text prompt. Must be a non-empty string.
            max_tokens (int): Maximum number of tokens to generate. Must be a positive integer
                not exceeding max_model_len.
        Returns:
            Request: A new Request object with state=PENDING, containing the prompt, token_ids,
                max_tokens, and a unique request_id (UUID).
        Raises:
            ValueError: If prompt is empty/non-string, or max_tokens is non-positive/exceeds max_model_len.
        '''
        
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if max_tokens > self.max_model_len:
            raise ValueError(f"max_tokens cannot exceed the model's maximum context length of {self.max_model_len}")
        
        token_ids = self.tokenizer.encode(text = prompt, return_tensor = False)[0]
        
        request_id = str(uuid.uuid4())
        
        return Request(request_id = request_id,
                       prompt = prompt,
                       token_ids = token_ids,
                       max_tokens = max_tokens)
        
