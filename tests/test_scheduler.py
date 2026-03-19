import logging # type: ignore

from llm_engine import Request, Tokenizer, BatchScheduler as Scheduler, load_asset_paths

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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
    
    scheduler = Scheduler(max_batch_size = 2)
    
    requests_list = []
    for idx in range(len(prompts)):
        
        request_id = str(idx).zfill(3)
        prompt = prompts[idx]
        token_ids = tokenizer.encode(prompt, return_tensor = False)[0]
        
        requests_list.append(Request(request_id = request_id,
                                prompt = prompt,
                                token_ids = token_ids,
                                max_tokens = max_tokens))
        
    # logger.info(f"Requests: {requests_list}")
    
    scheduler.add_request(requests_list[0])
    scheduler.add_request(requests_list[1])
    scheduler.add_request(requests_list[2])
    
    scheulded_requests = scheduler.schedule()
        
    scheduler.add_request(requests_list[3])
    scheduler.add_request(requests_list[4])
    
    scheduler.complete_request(request_id = scheulded_requests[0].request_id)
    
    
    for idx in range(len(prompts)):
        req = requests_list[idx]
        logger.info(f"Request IDs: {req.request_id}, Request Status: {req.state}")
        
    assert scheulded_requests[0].state == 'finished', f"Expected state 'finished', got {scheulded_requests[0].state}"
    assert scheulded_requests[1].state == 'running', f"Expected state 'running', got {scheulded_requests[1].state}"
    assert requests_list[2].state == 'pending', f"Expected state 'pending', got {requests_list[2].state}"