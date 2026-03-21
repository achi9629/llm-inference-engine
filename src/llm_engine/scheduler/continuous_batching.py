
"""
Continuous batching scheduler for the LLM inference engine.

Unlike static batching (BatchScheduler) which waits for an entire batch to 
finish before starting the next, continuous batching dynamically manages the 
active batch each step: finished requests are evicted and new pending requests 
fill the empty slots immediately.

This maximizes GPU utilization by keeping the batch full at all times, even 
when requests finish at different times.
"""

from typing import List

from .request_queue import RequestQueue
from .request import Request, RequestState

class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size: int) ->None:
        
        '''
        Description:
            Initialize the continuous batching scheduler.
        Args:
            max_batch_size: The maximum number of requests that can be processed in a batch simultaneously
        Returns:
            None
        '''
        
        self.request_queue = RequestQueue()
        self.max_batch_size = max_batch_size
        self.running_requests = {}
        self.completed_requests = {}
        
    def add_request(self, request: Request) -> None:
        
        '''
        Description:
            Add a new request to the scheduler.
        Args:
            request: The request object to be added to the scheduler
        Returns:
            None
        '''
        
        self.request_queue.add(request)
        
    def step(self) -> List[Request]:
        
        '''
        Description:
            Performs one scheduling step: evicts finished requests from the active batch,
            fills empty slots with pending requests from the queue, and returns the
            current active batch for the next generation iteration.
        Args:
            None
        Returns:
            A list of currently active requests in the batch after the scheduling step
        '''
        
        # 1. Remove finished requests
        finished_ids = [rid for rid, req in self.running_requests.items() if req.is_finished()]
        for rid in finished_ids:
            self.running_requests.pop(rid)
            
        # 2. Fill empty slots from queue
        num_empty_slots = self.max_batch_size - len(self.running_requests)
        new_requests = self.request_queue.get_batch(num_empty_slots)
        for req in new_requests:
            req.set_state(RequestState.RUNNING)
            self.running_requests[req.request_id] = req
            
        # 3. Return active batch
        return list(self.running_requests.values())
    
    def get_active_batch(self) -> List[Request]:
        
        '''
        Description:
            Get the current active batch of requests without modifying the scheduler state.
        Args:
            None
        Returns:
            A list of currently active requests in the batch
        '''
        
        return list(self.running_requests.values())
    
    def has_work(self) -> bool:
        
        '''
        Description:
            Check if there are any active or pending requests in the scheduler.
        Args:
            None
        Returns:
            True if there are active or pending requests, False otherwise
        '''
        
        return not self.request_queue.is_empty() or len(self.running_requests) > 0
    
    def complete_request(self, request_id: str) -> None:
        
        '''
        Description:
            Marks a request as completed and moves it from the running requests to the completed requests.
        Args:
            request_id (str): The ID of the request to be marked as completed.
        Returns:
            None
        '''
        
        if request_id in self.running_requests:
            req = self.running_requests.pop(request_id)
            req.set_state(RequestState.FINISHED)
            self.completed_requests[request_id] = req
        else:
            raise ValueError(f"Request ID {request_id} not found in running requests.")