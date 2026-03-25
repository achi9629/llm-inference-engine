
"""
Router module for the LLM inference engine serving layer.

Orchestrates the request lifecycle: validates and tokenizes the prompt
via RequestHandler, manages scheduling via ContinuousBatchingScheduler,
runs inference via InferenceEngine, and returns the final result.
"""

from .request_handler import RequestHandler
from ..inference.inference_engine import InferenceEngine
from ..scheduler.continuous_batching import ContinuousBatchingScheduler

class Router:
    def __init__(self, 
                 request_handler: RequestHandler,
                 scheduler: ContinuousBatchingScheduler,
                 engine: InferenceEngine
        ) -> None:
        
        '''
        Description:
            Initialize the Router with its three dependencies. The Router acts as
            the bridge between the HTTP layer and the internal scheduling/inference
            pipeline.
        Args:
            request_handler (RequestHandler): Validates and tokenizes incoming prompts.
            scheduler (ContinuousBatchingScheduler): Manages request lifecycle and batch scheduling.
            engine (InferenceEngine): Runs the model forward pass for text generation.
        Returns:
            None
        '''
        
        if not request_handler:
            raise ValueError("RequestHandler cannot be None")
        if not scheduler:
            raise ValueError("Scheduler cannot be None")
        if not engine:
            raise ValueError("InferenceEngine cannot be None")
        
        self.request_handler = request_handler
        self.scheduler = scheduler
        self.engine = engine
        
    def generate(self,
                 prompt: str,
                 max_tokens: int
        ) -> dict:
        
        '''
        Description:
            Process a single generation request end-to-end. Validates the prompt,
            submits it to the scheduler, runs inference via the engine, marks the
            request as completed, and returns the result.
        Args:
            prompt (str): The input text to generate from.
            max_tokens (int): Maximum number of tokens to generate.
        Returns:
            dict: A dictionary containing request_id, prompt, generated_text,
                token_count, and stop_reason.
        '''
        
        request = self.request_handler.handle(prompt = prompt, 
                                              max_tokens = max_tokens)
        
        
        self.scheduler.add_request(request)
        
        self.scheduler.step() # instead of: batch = self.scheduler.step()
        
        result = self.engine.generate(input_text = prompt, max_tokens = max_tokens)
        
        self.scheduler.complete_request(request_id = request.request_id)
        
        
        return {
            "request_id": request.request_id,
            "prompt": prompt,
            "generated_text": result['generated_text'],
            "token_count": result['token_count'],
            "stop_reason": result['stop_reason']
        }