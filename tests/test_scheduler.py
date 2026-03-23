import torch

from llm_engine import Request, Tokenizer, BatchScheduler as Scheduler, load_asset_paths
from llm_engine import ContinuousBatchingScheduler, MemoryAllocator, BlockTable, PagedKVCache

def test_scheduler():
    
    config, _ = load_asset_paths()
    tokenizer = Tokenizer(config)
    max_tokens = 50
    prompts = [
                "What is the capital of France?","In a galaxy far far away",
                "The key to understanding transformers is",
                "Artificial intelligence will change",
                "The future of quantum computing",
                "Once upon a time in a land",
                "The theory of relativity explains",
    ]
    
    max_batch_size = 2
    scheduler = Scheduler(max_batch_size = max_batch_size)
    
    requests_list = []
    for idx in range(len(prompts)):
        
        request_id = str(idx).zfill(3)
        prompt = prompts[idx]
        token_ids = tokenizer.encode(prompt, return_tensor = False)[0]
        
        requests_list.append(Request(request_id = request_id,
                                prompt = prompt,
                                token_ids = token_ids,
                                max_tokens = max_tokens))
    
    scheduler.add_request(requests_list[0])
    scheduler.add_request(requests_list[1])
    scheduler.add_request(requests_list[2])
    
    scheulded_requests = scheduler.schedule()
        
    scheduler.add_request(requests_list[3])
    scheduler.add_request(requests_list[4])
    
    scheduler.complete_request(request_id = scheulded_requests[0].request_id)
        
    assert scheulded_requests[0].state == 'finished', f"Expected state 'finished', got {scheulded_requests[0].state}"
    assert scheulded_requests[1].state == 'running', f"Expected state 'running', got {scheulded_requests[1].state}"
    assert requests_list[2].state == 'pending', f"Expected state 'pending', got {requests_list[2].state}"
    
def test_continuous_batching_step():
    
    config, _ = load_asset_paths()
    tokenizer = Tokenizer(config)
    max_tokens = 50
    prompts = [
                "What is the capital of France?","In a galaxy far far away",
                "The key to understanding transformers is",
                "Artificial intelligence will change",
                "The future of quantum computing",
                "Once upon a time in a land",
                "The theory of relativity explains",
    ]
    
    max_batch_size = 2
    scheduler = ContinuousBatchingScheduler(max_batch_size = max_batch_size)
    
    requests_list = []
    for idx in range(len(prompts)):
        
        request_id = str(idx).zfill(3)
        prompt = prompts[idx]
        token_ids = tokenizer.encode(prompt, return_tensor = False)[0]
        
        requests_list.append(Request(request_id = request_id,
                                prompt = prompt,
                                token_ids = token_ids,
                                max_tokens = max_tokens))
        
    
    scheduler.add_request(requests_list[0])
    scheduler.add_request(requests_list[1])
    scheduler.add_request(requests_list[2])
    scheduler.add_request(requests_list[3])
    
    batch = scheduler.step()
    
    assert len(batch) == max_batch_size, f"Expected batch size {max_batch_size}, got {len(batch)}"
    assert batch[0].state == 'running', f"Expected state 'running', got {batch[0].state}"
    assert batch[1].state == 'running', f"Expected state 'running', got {batch[1].state}"
    assert scheduler.request_queue.size() == 2, f"Expected queue size 2, got {scheduler.request_queue.size()}"
    
    
    scheduler.complete_request(request_id = requests_list[0].request_id)
    batch = scheduler.step()
    
    assert len(batch) == max_batch_size, f"Expected batch size {max_batch_size}, got {len(batch)}"
    assert requests_list[0].request_id not in scheduler.running_requests, f"Expected request {requests_list[0].request_id} not be in running requests"
    assert requests_list[0].state == 'finished', f"Expected state 'finished', got {requests_list[0].state}"
    assert requests_list[1].request_id in scheduler.running_requests, f"Expected request {requests_list[1].request_id} to be in running requests"
    assert requests_list[1].state == 'running', f"Expected state 'running', got {requests_list[1].state}"
    assert requests_list[2].request_id in scheduler.running_requests, f"Expected request {requests_list[2].request_id} to be in running requests"
    assert requests_list[2].state == 'running', f"Expected state 'running', got {requests_list[2].state}"
    assert scheduler.request_queue.size() == 1, f"Expected queue size 1, got {scheduler.request_queue.size()}"
    
    scheduler.complete_request(request_id = requests_list[1].request_id)
    scheduler.complete_request(request_id = requests_list[2].request_id)
    batch = scheduler.step()
    
    assert len(batch) == 1, f"Expected batch size 1, got {len(batch)}"
    assert requests_list[1].request_id not in scheduler.running_requests, f"Expected request {requests_list[1].request_id} not be in running requests"
    assert requests_list[1].state == 'finished', f"Expected state 'finished', got {requests_list[1].state}"
    assert requests_list[2].request_id not in scheduler.running_requests, f"Expected request {requests_list[2].request_id} not be in running requests"
    assert requests_list[2].state == 'finished', f"Expected state 'finished', got {requests_list[2].state}"
    assert requests_list[3].request_id in scheduler.running_requests, f"Expected request {requests_list[3].request_id} to be in running requests"
    assert requests_list[3].state == 'running', f"Expected state 'running', got {requests_list[3].state}"
    assert scheduler.request_queue.size() == 0, f"Expected queue size 0, got {scheduler.request_queue.size()}"
    
    scheduler.complete_request(request_id = requests_list[3].request_id)
    batch = scheduler.step()
    
    assert len(batch) == 0, f"Expected batch size 0, got {len(batch)}"
    assert scheduler.has_work() == False, f"Expected has_work to be False, got {scheduler.has_work()}"
    assert requests_list[3].request_id not in scheduler.running_requests, f"Expected request {requests_list[3].request_id} not be in running requests"
    assert requests_list[3].state == 'finished', f"Expected state 'finished', got {requests_list[3].state}"
    
def test_continuous_batching_with_paged_cache():

    config, _ = load_asset_paths()
    tokenizer = Tokenizer(config)
    max_tokens = 50
    num_blocks = 16
    n_layers = 2
    n_heads = 4
    block_size = 4
    head_dim = 8
    max_batch_size = 2
    
    prompts = [
                "Explain how a jet engine works very simply.",
                "Write a short poem about a lonely robot.",
                "Create a healthy meal plan for one week.",
                "Describe the history of the ancient Roman Empire.",
                "Give me five tips for better sleep tonight.",
                "List the most popular tourist spots in Paris.",
    ]
    
    allocator = MemoryAllocator(num_blocks = num_blocks)
    block_table = BlockTable(allocator = allocator, block_size = 4)
    paged_kv_cache = PagedKVCache(num_blocks = num_blocks,
                                  n_layers = n_layers,
                                  n_heads = n_heads,
                                  block_size = block_size,
                                  head_dim = head_dim,
                                  dtype = torch.float32,
                                  device = 'cpu')
    
    scheduler = ContinuousBatchingScheduler(max_batch_size = max_batch_size,
                                            block_table = block_table,
                                            paged_kv_cache = paged_kv_cache)
    
    
    requests_list = []
    for idx in range(len(prompts)):
        
        request_id = str(idx).zfill(3)
        prompt = prompts[idx]
        token_ids = tokenizer.encode(prompt, return_tensor = False)[0][ : 8]
        
        requests_list.append(Request(request_id = request_id,
                                prompt = prompt,
                                token_ids = token_ids,
                                max_tokens = max_tokens))
        
    for req in requests_list[:3]:
        scheduler.add_request(req)
        
    batch = scheduler.step()
    
    assert len(batch) == max_batch_size, f"Expected batch size {max_batch_size}, got {len(batch)}"
    assert allocator.num_free_blocks == 12, f"Expected 12 free blocks, got {allocator.num_free_blocks}"
    for b in batch:
        assert len(block_table.get_block_ids(b.request_id)) == 2, f"Expected 2 blocks allocated for request {b.request_id}, got {len(block_table.get_block_ids(b.request_id))}"
        
    scheduler.complete_request(request_id = batch[0].request_id)
    batch = scheduler.step()
    
    assert len(batch) == max_batch_size, f"Expected batch size {max_batch_size}, got {len(batch)}"
    assert allocator.num_free_blocks == 12, f"Expected 12 free blocks, got {allocator.num_free_blocks}"
    assert requests_list[0].request_id not in block_table.block_table, f"Expected request {requests_list[0].request_id} to be deallocated from block table"
    
    scheduler.complete_request(request_id = batch[0].request_id)
    scheduler.complete_request(request_id = batch[1].request_id)
    batch = scheduler.step()
    
    assert len(batch) == 0, f"Expected batch size 0, got {len(batch)}"
    assert scheduler.has_work() == False, f"Expected has_work to be False, got {scheduler.has_work()}"
    assert allocator.num_free_blocks == num_blocks, f"Expected all blocks to be free, got {allocator.num_free_blocks}"
    for req in requests_list[:3]:
        assert req.request_id not in block_table.block_table, f"Expected request {req.request_id} to be deallocated from block table"
        assert req.is_finished() == True, f"Expected request {req.request_id} to be finished"