"""
Request module for the LLM inference engine.

Defines the Request dataclass that tracks a single inference request 
through its lifecycle: PENDING → RUNNING → FINISHED/FAILED.

Each Request holds the original prompt, tokenized input, generated tokens,
and metadata (state, timestamps) needed by the scheduler and server.
"""

from time import time
from typing import List

class RequestState:
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    
class Request:
    def __init__(self,
                 request_id: str,
                 prompt: str,
                 token_ids: List[int],
                 max_tokens: int,
                 ) -> None:
        
        '''
        Description: 
            Initializes a new Request object with the given parameters. 
        Args:
            request_id (str): Unique identifier for the request.
            prompt (str): The original text prompt for the inference.
            token_ids (List[int]): List of token IDs representing the input prompt.
            max_tokens (int): Maximum number of tokens to generate in response.
        Returns:
            None
        '''
        
        self.request_id = request_id
        self.prompt = prompt
        self.token_ids = token_ids
        self.max_tokens = max_tokens
        self.generated_tokens = []
        self.state = RequestState.PENDING
        self.created_at = time()
        
    def add_generated_token(self, token_id: int) -> None:
        
        '''
        Description: 
            Adds a generated token ID to the list of generated tokens for this request.
        Args:
            token_id (int): The token ID of the newly generated token to add.
        Returns:
            None
        '''
        
        self.generated_tokens.append(token_id)
        
    def set_state(self, state: str) -> None:
        
        '''
        Description: 
            Updates the state of the request to the given state.
        Args:
            state (str): The new state to set for the request. Should be one of the values defined in RequestState.
        Returns:
            None
        '''
        
        self.state = state
        
    def is_finished(self) -> bool:
        
        '''
        Description: 
            Checks if the request has reached a terminal state (either FINISHED or FAILED).
        Args:
            None
        Returns:
            bool: True if the request is in a terminal state, False otherwise.
        '''
        
        return self.state in [RequestState.FINISHED, RequestState.FAILED]
    
    def num_generated_tokens(self) -> int:
        
        '''
        Description: 
            Returns the number of tokens that have been generated so far for this request.
        Args:
            None
        Returns:
            int: The number of generated tokens.
        '''
        
        return len(self.generated_tokens)
    