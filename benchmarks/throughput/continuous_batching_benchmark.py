import torch, logging

from llm_engine import load_model, load_asset_paths, Tokenizer, InferenceEngine
from llm_engine import Request, RequestState, ContinuousBatchingScheduler
from benchmarks.metrics import BenchmarkMetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

def run_static_batching_simulation(is_kv_cache_enabled: bool = True) -> None:
    
    config, model_cfg = load_asset_paths()
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sampling_method = 'greedy'
    
    logger.info(f"KV Cache Enabled: {is_kv_cache_enabled}")
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = model.num_parameters(),
                               dtype = next(model.parameters()).dtype)
    
    max_batch_size = 4
    max_tokens = 50
    engine = InferenceEngine(model = model, 
                    device = device, 
                    tokenizer = tokenizer,
                    eos_token_id = model_cfg['eos_token_id'],
                    sampling_method = sampling_method,
                    is_kv_cache_enabled = is_kv_cache_enabled,
                    max_tokens_for_kv_cache = model_cfg['n_ctx'],
                    batch_size = max_batch_size,
                    model_cfg = model_cfg
        )
    
    # Warmup (not timed)
    engine.generate(["warmup"] * max_batch_size, max_tokens = 10)
    
    
    prompts = [["Deep learning has revolutionized the field of",
                "The meaning of life is",
                "In a galaxy far far away",
                "The key to understanding transformers is"],
                ["Artificial intelligence will change",
                "The future of quantum computing",
                "Once upon a time in a land",
                "The theory of relativity explains"],
                ["Deep learning has revolutionized the field of",
                "The meaning of life is",
                "In a galaxy far far away",
                "The key to understanding transformers is"],
                ["Artificial intelligence will change",
                "The future of quantum computing",
                "Once upon a time in a land",
                "The theory of relativity explains",],
                ["Deep learning has revolutionized the field of",
                "The meaning of life is",
                "In a galaxy far far away",
                "The key to understanding transformers is",],
                ["Artificial intelligence will change",
                "The future of quantum computing",
                "Once upon a time in a land",
                "The theory of relativity explains",]]
    
    results = []
    for prompt in prompts:
        logger.info(f"Running benchmark for batch of prompts: {prompt}")
        metrics.start()
        output = engine.generate(input_text = prompt, max_tokens = max_tokens)
        total_tokens = sum(r['token_count'] for r in output) if isinstance(output, list) else output['token_count']
        metrics.stop(num_tokens = total_tokens)
        
        result = metrics.result()
        result['total_tokens'] = total_tokens
        results.append(result)
        
    # Print summary table
    header = (
        f"{'Total Tokens':<14} {'Latency (s)':<14} "
        f"{'Tok/s':<9} {'Peak Mem (MB)':<15} {'GPU Util (%)':<15} {'MFU (%)':<10}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        row = (
            f"{r['total_tokens']:<14} "
            f"{r['latency_sec']:<14.3f} {r['tokens_per_sec']:<9.1f} "
            f"{r['peak_memory_mb']:<15.1f} "
            f"{r['gpu_snapshot']['gpu_utilization_percent']:<15.1f} "
            f"{r['MFU_percent']:<10.2f}"
        )
        logger.info(row)
        
def run_continuous_batching_simulation(is_kv_cache_enabled: bool = True) -> None:
    
    config, model_cfg = load_asset_paths()
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sampling_method = 'greedy'
    
    logger.info(f"KV Cache Enabled: {is_kv_cache_enabled}")
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = model.num_parameters(),
                               dtype = next(model.parameters()).dtype)
    
    max_batch_size = 4
    max_tokens = 50
    engine = InferenceEngine(model = model, 
                    device = device, 
                    tokenizer = tokenizer,
                    eos_token_id = model_cfg['eos_token_id'],
                    sampling_method = sampling_method,
                    is_kv_cache_enabled = is_kv_cache_enabled,
                    max_tokens_for_kv_cache = model_cfg['n_ctx'],
                    batch_size = max_batch_size,
                    model_cfg = model_cfg
        )
    
    # Warmup (not timed)
    engine.generate(["warmup"] * max_batch_size, max_tokens = 10)

    all_prompts = [
        "Deep learning has revolutionized the field of",
        "The meaning of life is",
        "In a galaxy far far away",
        "The key to understanding transformers is",
        "Artificial intelligence will change",
        "The future of quantum computing",
        "Once upon a time in a land",
        "The theory of relativity explains",
        "Deep learning has revolutionized the field of",
        "The meaning of life is",
        "In a galaxy far far away",
        "The key to understanding transformers is",
        "Artificial intelligence will change",
        "The future of quantum computing",
        "Once upon a time in a land",
        "The theory of relativity explains",
        "Deep learning has revolutionized the field of",
        "The meaning of life is",
        "In a galaxy far far away",
        "The key to understanding transformers is",
        "Artificial intelligence will change",
        "The future of quantum computing",
        "Once upon a time in a land",
        "The theory of relativity explains",
    ]
    
    scheduler = ContinuousBatchingScheduler(max_batch_size = max_batch_size)
    
    # Add all requests to queue
    for idx, prompt in enumerate(all_prompts):
        # Add all requests to queue
        token_ids = tokenizer.encode(prompt, return_tensor=False)[0]
        req = Request(request_id=str(idx),
                      prompt = prompt,
                      token_ids = token_ids,
                      max_tokens = max_tokens
                )
        
        scheduler.add_request(req)
        
    # Measure total time
    metrics.start()
    total_tokens = 0
    while scheduler.has_work():
        batch = scheduler.step()  # fills slots with pending requests
        if not batch:
            break
        prompts_batch = [req.prompt for req in batch]
        output = engine.generate(input_text=prompts_batch, max_tokens=max_tokens)
        tokens = sum(r['token_count'] for r in output) if isinstance(output, list) else output['token_count']
        total_tokens += tokens
        
        # Mark all as finished (since generate() runs to completion)
        for req in batch:
            req.set_state(RequestState.FINISHED)
    metrics.stop(num_tokens=total_tokens)
    
    result = metrics.result()
    result['total_tokens'] = total_tokens
    
    header = (
        f"{'Total Tokens':<14} {'Latency (s)':<14} "
        f"{'Tok/s':<9} {'Peak Mem (MB)':<15} {'GPU Util (%)':<15} {'MFU (%)':<10}"
    )
    
    logger.info(header)
    logger.info("-" * len(header))
    row = (
        f"{result['total_tokens']:<14} "
        f"{result['latency_sec']:<14.3f} {result['tokens_per_sec']:<9.1f} "
        f"{result['peak_memory_mb']:<15.1f} "
        f"{result['gpu_snapshot']['gpu_utilization_percent']:<15.1f} "
        f"{result['MFU_percent']:<10.2f}"
    )
    logger.info(row)
        
if __name__ == "__main__":
    
    run_static_batching_simulation()
    run_continuous_batching_simulation()