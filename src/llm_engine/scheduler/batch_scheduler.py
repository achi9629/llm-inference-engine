
"""
Batch scheduler module for the LLM inference engine.

Manages the lifecycle of inference requests by pulling pending requests
from the queue, grouping them into batches (up to max_batch_size), and
tracking their state transitions: PENDING → RUNNING → FINISHED/FAILED.

Acts as the central coordinator between the request queue and the 
inference engine.
"""

from typing import List

from .request_queue import RequestQueue
from .request import Request, RequestState

class BatchScheduler:
    def __init__(self, max_batch_size: int) -> None:
        
        '''
        Description:
            Initializes the BatchScheduler with a specified maximum batch size. It sets up the 
            request queue, and initializes dictionaries to track running and completed requests.
        Args:
            max_batch_size (int): The maximum number of requests that can be scheduled in a single batch.
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
            Adds a new request to the request queue. The request is initially marked as pending.
        Args:
            request (Request): The request to be added to the queue.
        Returns:
            None
        '''
        
        self.request_queue.add(request)
        
    def schedule(self) -> List[Request]:
        
        '''
        Description:
            Schedules a batch of requests from the request queue. It retrieves a batch of 
            requests up to the maximum batch size, marks them as running, and moves them 
            to the running requests dictionary.
        Args:
            None
        Returns:
            List[Request]: A list of requests that have been scheduled to run.
        '''
        
        batch = self.request_queue.get_batch(self.max_batch_size)
        for req in batch:
            req.set_state(RequestState.RUNNING)
            self.running_requests[req.request_id] = req
        return batch
    
    def get_running_requests(self) -> List[Request]:
        
        '''
        Description:
            Returns a list of currently running requests.
        Args:
            None
        Returns:
            List[Request]: A list of currently running requests.
        '''
        
        return list(self.running_requests.values())
    
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