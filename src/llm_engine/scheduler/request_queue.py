
"""
Request queue module for the LLM inference engine.

A FIFO queue that holds pending Request objects waiting to be scheduled
for generation. The batch scheduler pulls requests from this queue 
and groups them into batches for inference.
"""

from collections import deque

from .request import Request

class RequestQueue :
    def __init__(self) -> None:
        
        '''
        Description:
            A simple queue to hold incoming requests for processing. It supports adding requests, 
            popping the next request, and retrieving a batch of requests for processing. The queue 
            is implemented using a deque for efficient appending and popping operations.
        Args:
            None
        Returns:
            None
        '''
    
        self.queue = deque()
        
    def add(self, request: Request) -> None:
        
        '''
        Description:
            Add a new request to the end of the queue. This method is called when a new request 
            arrives and needs to be scheduled for processing. The request is appended to the end 
            of the deque, ensuring that requests are processed in the order they were received.
        Args:
            request (Request): The request object to be added to the queue.
        Returns:
            None
        '''
        
        self.queue.append(request)
        
    def pop(self) -> Request | None:
        
        '''
        Description:
            Remove and return the next request from the front of the queue. This method is called 
            by the batch scheduler when it is ready to process the next request. If the queue is 
            empty, it returns None.
        Args:
            None
        Returns:
            Request | None: The next request in the queue, or None if the queue is empty.
        '''
        
        if self.queue:
            return self.queue.popleft()
        return None
    
    def get_batch(self, max_batch_size: int) -> list[Request]:
        
        '''
        Description:
            Retrieve a batch of requests from the front of the queue, up to the specified maximum 
            batch size. This method is called by the batch scheduler when it is ready to process a 
            batch of requests. It pops requests from the front of the queue until it reaches the 
            maximum batch size or the queue is empty.
        Args:
            max_batch_size (int): The maximum number of requests to include in the batch.
        Returns:
            list[Request]: A list of requests retrieved from the queue, up to the specified batch size.
        '''
    
        batch = []
        while self.queue and len(batch) < max_batch_size:
            batch.append(self.queue.popleft())
        return batch
    
    def size(self) -> int:
        
        '''
        Description:
            Get the current number of requests in the queue. This method can be used by the batch 
            scheduler to determine how many requests are waiting to be processed and to decide when 
            to trigger batch processing.
        Args:
            None
        Returns:
            int: The number of requests currently in the queue.
        '''
        
        return len(self.queue)
    
    def is_empty(self) -> bool:
        
        '''
        Description:
            Check if the queue is empty. This method can be used by the batch scheduler to quickly 
            determine if there are any requests waiting to be processed.
        Args:
            None
        Returns:
            bool: True if the queue is empty, False otherwise.
        '''
        
        return len(self.queue) == 0