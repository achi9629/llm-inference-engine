
"""
Continuous batching scheduler for the LLM inference engine.

Unlike static batching (BatchScheduler) which waits for an entire batch to 
finish before starting the next, continuous batching dynamically manages the 
active batch each step: finished requests are evicted and new pending requests 
fill the empty slots immediately.

This maximizes GPU utilization by keeping the batch full at all times, even 
when requests finish at different times.

Optionally integrates with PagedKVCache via BlockTable:
    - On evict: frees blocks and resets paged cache for finished sequences
    - On fill: allocates initial blocks for new sequences
    - Write/read during decode happens in the attention layer, not here
"""

import math

from .request_queue import RequestQueue
from .request import Request, RequestState
from ..cache.block_table import BlockTable
from ..cache.paged_kv_cache import PagedKVCache

class ContinuousBatchingScheduler:
    def __init__(self, 
                 max_batch_size: int,
                 block_table: BlockTable | None = None,
                 paged_kv_cache: PagedKVCache | None = None,
        ) ->None:
        
        '''
        Description:
            Initialize the continuous batching scheduler with optional paged memory management.
            When block_table and paged_kv_cache are provided, the scheduler automatically
            allocates blocks for new requests and frees blocks when requests finish.
        Args:
            max_batch_size (int): Maximum number of requests in the active batch.
            block_table (BlockTable | None): Block table for paged KV cache management. None disables paging.
            paged_kv_cache (PagedKVCache | None): Paged KV cache for block-level reset on eviction. None disables paging.
        Returns:
            None
        '''
        
        self.request_queue = RequestQueue()
        self.max_batch_size = max_batch_size
        self.block_table = block_table
        self.paged_kv_cache = paged_kv_cache
        self.running_requests = {}
        self.completed_requests = {}
        
    def add_request(self, 
                    request: Request
        ) -> None:
        
        '''
        Description:
            Add a new request to the scheduler.
        Args:
            request: The request object to be added to the scheduler
        Returns:
            None
        '''
        
        self.request_queue.add(request)
        
    def step(self) -> list[Request]:
        
        '''
        Description:
            Performs one scheduling step:
                Phase 1 — Evict finished requests from running batch.
                    If paged: free blocks via block_table, reset via paged_kv_cache.
                Phase 2 — Fill empty slots with pending requests from queue.
                    If paged: add sequence to block_table, allocate initial blocks (ceil(token_ids / block_size)).
                Phase 3 — Return current active batch.
        Args:
            None
        Returns:
            list[Request]: Currently active requests after the scheduling step.
        '''
        
        # 1. Remove finished requests
        self.finished_ids = [rid for rid, req in self.running_requests.items() if req.is_finished()]
        for rid in self.finished_ids:
            if self.block_table:
                block_ids = self.block_table.get_block_ids(seq_id = rid)
                self.paged_kv_cache.reset_blocks(block_ids)
                self.block_table.free_sequence(rid)
            self.running_requests.pop(rid)
            
        # 2. Fill empty slots from queue
        num_empty_slots = self.max_batch_size - len(self.running_requests)
        new_requests = self.request_queue.get_batch(num_empty_slots)
        for req in new_requests:
            req.set_state(RequestState.RUNNING)
            self.running_requests[req.request_id] = req
            if self.block_table:
                self.block_table.add_sequence(req.request_id)
                num_initial_blocks = math.ceil(len(req.token_ids) / self.block_table.block_size)
                self.block_table.allocate_blocks(req.request_id, num_initial_blocks)
            
        # 3. Return active batch
        return list(self.running_requests.values())
    
    def get_active_batch(self) -> list[Request]:
        
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
            req = self.running_requests[request_id]
            req.set_state(RequestState.FINISHED)
            self.completed_requests[request_id] = req
        else:
            raise ValueError(f"Request ID {request_id} not found in running requests.")