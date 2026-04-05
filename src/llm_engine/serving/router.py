
"""
Router module for the LLM inference engine serving layer.

Orchestrates the request lifecycle: validates and tokenizes the prompt
via RequestHandler, manages scheduling via ContinuousBatchingScheduler,
runs inference via InferenceEngine, and returns the final result.

Supports two modes:
- Synchronous (async_mode=False): processes one request at a time inline.
- Asynchronous (async_mode=True): accepts concurrent requests via asyncio futures
  and processes them in batches through a background generation loop. The scheduler
  groups up to max_batch_size requests, which are passed as a list to
  engine.generate() for a single batched GPU forward pass.

Error handling: if engine.generate() fails during batched inference, all futures
in the affected batch are resolved with the exception, and the loop continues
processing subsequent batches.
"""

import asyncio, logging

from .request_handler import RequestHandler
from ..inference.inference_engine import InferenceEngine
from ..scheduler.continuous_batching import ContinuousBatchingScheduler

logger = logging.getLogger(__name__)

class Router:
    def __init__(self, 
                 request_handler: RequestHandler,
                 scheduler: ContinuousBatchingScheduler,
                 engine: InferenceEngine,
                 async_mode: bool = False,
        ) -> None:
        
        '''
        Description:
            Initialize the Router with its three dependencies and configure the
            operating mode. The Router acts as the bridge between the HTTP layer
            and the internal scheduling/inference pipeline.
        Args:
            request_handler (RequestHandler): Validates and tokenizes incoming prompts.
            scheduler (ContinuousBatchingScheduler): Manages request lifecycle and batch scheduling.
            engine (InferenceEngine): Runs the model forward pass for text generation.
            async_mode (bool): If True, enables concurrent request handling via a background
                generation loop and asyncio futures. If False, processes requests inline
                one at a time. Default is False.
        Raises:
            ValueError: If request_handler, scheduler, or engine is None.
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
        
        self.async_mode = async_mode
        self._pending_futures = {} # Maps request_id to asyncio.Future for pending requests
        self._lock = asyncio.Lock() # To protect access to _pending_futures
        self._loop_task = None # Task for the scheduler loop, will hold the reference to the background task running the scheduler
        
    def start(self) -> None:
        
        '''
        Description:
            Start the Router. In async mode, launches the background _generation_loop
            as an asyncio Task that continuously drains the scheduler and runs batched
            inference. In sync mode, this is a no-op.
        Args:
            None
        Returns:
            None
        '''
        
        if not self.async_mode:
            return
        self.loop = asyncio.get_event_loop()
        self._loop_task = self.loop.create_task(self._generation_loop())
        
    def stop(self) -> None:
        
        '''
        Description:
            Stop the Router's background generation loop. Cancels the asyncio Task
            if it is running. Safe to call even if the loop was never started.
        Args:
            None
        Returns:
            None
        '''
        
        if self._loop_task:
            self._loop_task.cancel()
            self._loop_task = None
        
    async def generate(self,
                 prompt: str,
                 max_tokens: int
        ) -> dict:
        
        '''
        Description:
            Process a single generation request end-to-end.

            Sync mode: validates the prompt, submits it to the scheduler, runs
            inference via the engine inline (single prompt), marks the request as
            completed, and returns the result immediately.

            Async mode: validates the prompt, submits it to the scheduler, creates
            an asyncio Future, and awaits the result. The background _generation_loop
            batches multiple pending requests and runs them through a single
            engine.generate() call. The Future is resolved when the batch completes.
            If inference fails, the Future receives the exception.
        Args:
            prompt (str): The input text to generate from.
            max_tokens (int): Maximum number of tokens to generate.
        Returns:
            dict: A dictionary containing request_id, prompt, generated_text,
                token_count, and stop_reason.
        Raises:
            ValueError: If prompt is empty/non-string, or max_tokens is invalid.
            Exception: Re-raised from engine.generate() if batched inference fails
                (async mode only, propagated via Future.set_exception).
        '''
        
        # Synchronous path for backward compatibility
        if not self.async_mode:
            
            request = self.request_handler.handle(prompt = prompt, 
                                                  max_tokens = max_tokens)
        
            self.scheduler.add_request(request)
            
            self.scheduler.step() # instead of: batch = self.scheduler.step()
            
            result = self.engine.generate(input_text = prompt, max_tokens = max_tokens)
        
            self.scheduler.complete_request(request_id = request.request_id)
            
        else:
            
            request = self.request_handler.handle(prompt = prompt, 
                                                  max_tokens = max_tokens)
            
            async with self._lock:
                self.scheduler.add_request(request)
                
            future = self.loop.create_future()
            
            self._pending_futures[request.request_id] = future
        
            result = await future
                
        return {
                "request_id": request.request_id,
                "prompt": prompt,
                "generated_text": result['generated_text'],
                "token_count": result['token_count'],
                "stop_reason": result['stop_reason']
            }
        
    async def _generation_loop(self):
        
        '''
        Description:
            Background coroutine that runs continuously in async mode. On each
            iteration, acquires the lock, checks if the scheduler has pending
            requests, and if so, steps the scheduler to get the active batch.

            Collects all prompts and max_tokens from the batch and calls
            engine.generate() once with the full list for batched GPU inference.
            After generation, marks each request as complete and resolves the
            corresponding asyncio Futures to wake up waiting generate() callers.

            If engine.generate() raises an exception, logs the error and resolves
            all futures in the affected batch with set_exception() so clients
            receive a 500 error instead of hanging. The loop continues processing
            subsequent batches.

            When idle, sleeps for 10ms to avoid busy-spinning and releases the
            lock so new requests can be added.
        Args:
            None
        Returns:
            None (runs indefinitely until the task is cancelled).
        '''
        
        while True:
            has_work = False
            async with self._lock:
                
                if self.scheduler.has_work():
                    has_work = True
                    batch = self.scheduler.step()
                    
                    if batch:
                        try:
                            prompts = [request.prompt for request in batch]
                            max_tokens_list = [request.max_tokens for request in batch]
                            results = self.engine.generate(input_text = prompts, 
                                                        max_tokens = max_tokens_list)
                            
                            for request, result in zip(batch, results):
                                
                                self.scheduler.complete_request(request_id = request.request_id)
                                
                                result_dict = {
                                            "request_id": request.request_id,
                                            "prompt": request.prompt,
                                            "generated_text": result['generated_text'],
                                            "token_count": result['token_count'],
                                            "stop_reason": result['stop_reason']
                                        }
                                
                                future = self._pending_futures.pop(request.request_id, None)
                                
                                if future and not future.done():
                                    future.set_result(result_dict)
                        except Exception as e:
                            
                            logger.error("Generation loop error: %s", e, exc_info=True)
                            
                            for request in batch:
                                try:
                                    self.scheduler.complete_request(request_id=request.request_id)
                                except Exception:
                                    pass
                                future = self._pending_futures.pop(request.request_id, None)
                                if future and not future.done():
                                    future.set_exception(e)
                        
            if not has_work:
                
                await asyncio.sleep(0.01) # Sleep briefly to avoid busy waiting
        