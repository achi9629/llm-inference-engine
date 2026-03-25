from .request_handler import RequestHandler
from ..inference.inference_engine import InferenceEngine
from ..scheduler.continuous_batching import ContinuousBatchingScheduler

class Router:
    def __init__(self, 
                 request_handler: RequestHandler,
                 scheduler: ContinuousBatchingScheduler,
                 engine: InferenceEngine
        ) -> None:
        
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