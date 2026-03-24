import torch, logging, math
from tqdm import tqdm

from llm_engine import load_model, load_asset_paths, Tokenizer, InferenceEngine
from benchmarks.metrics import BenchmarkMetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

def run_throughput_peak_gpu_vs_block_benchmark(cache_type: str,
                                               num_blocks: int,
                                               block_size: int,
    ) -> None:
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    sampling_method = 'greedy'
    batch_size = 4
    max_tokens = 50
    
    logger.info(f"cache_type: {cache_type}, num_blocks: {num_blocks}, block_size: {block_size}")
    
    engine = InferenceEngine(model = model,
                             device = device,
                             tokenizer = tokenizer,
                             eos_token_id = model_cfg['eos_token_id'],
                             sampling_method = sampling_method,
                             is_kv_cache_enabled = True,
                             max_tokens_for_kv_cache = model_cfg['n_ctx'],
                             batch_size = batch_size,
                             model_cfg = model_cfg,
                             cache_type = cache_type,
                             num_blocks = num_blocks,
                             block_size = block_size,
                )
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = model.num_parameters(),
                               dtype = next(model.parameters()).dtype
                )
    
    
    # Warmup
    engine.generate(["Hello"] * batch_size, max_tokens = 2)
    
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
    for prompt in tqdm(prompts):
        
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

def run_throughput_peak_gpu_vs_batch_size_benchmark(cache_type: str, 
                                                    num_blocks: int, 
                                                    block_size: int,
                                                    batch_sizes: list[int],
        ) -> None:
        
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    sampling_method = 'greedy'
    max_tokens = 50
    
    logger.info(f"cache_type: {cache_type}, num_blocks: {num_blocks}, block_size: {block_size}")
    
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = model.num_parameters(),
                               dtype = next(model.parameters()).dtype
                )
    
    with open('assets/files/prompt_1024.txt', 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
        
    results = []
    for bs in tqdm(batch_sizes):
        
        batch_prompts = prompts[:bs]
        
        engine = InferenceEngine(model = model,
                             device = device,
                             tokenizer = tokenizer,
                             eos_token_id = model_cfg['eos_token_id'],
                             sampling_method = sampling_method,
                             is_kv_cache_enabled = True,
                             max_tokens_for_kv_cache = model_cfg['n_ctx'],
                             batch_size = bs,
                             model_cfg = model_cfg,
                             cache_type = cache_type,
                             num_blocks = num_blocks,
                             block_size = block_size,
                )
    
        # Warmup
        engine.generate(batch_prompts, max_tokens = 2)
        
        # Timed run
        metrics.start()
        output = engine.generate(batch_prompts, max_tokens)
        total_tokens = sum(r['token_count'] for r in output) if isinstance(output, list) else output['token_count']
        metrics.stop(num_tokens=total_tokens)
        
        result = metrics.result()
        result['batch_size'] = bs
        result['total_tokens'] = total_tokens
        results.append(result)
        
        # logger.info(f"Batch Size: {bs}, Total Tokens: {total_tokens}")
        # metrics.summary()
        # logger.info("")
        
    # Print summary table
    header = (
        f"{'Batch Size':<12} {'Total Tokens':<14} {'Latency (s)':<14} "
        f"{'Tok/s':<9} {'Peak Mem (MB)':<15} {'GPU Util (%)':<15} {'MFU (%)':<10}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        row = (
            f"{r['batch_size']:<12} {r['total_tokens']:<14} "
            f"{r['latency_sec']:<14.3f} {r['tokens_per_sec']:<9.1f} "
            f"{r['peak_memory_mb']:<15.1f} "
            f"{r['gpu_snapshot']['gpu_utilization_percent']:<15.1f} "
            f"{r['MFU_percent']:<10.2f}"
        )
        logger.info(row)
        
def run_throughput_peak_gpu_vs_max_tokens_benchmark(cache_type: str, 
                                                    num_blocks: int, 
                                                    block_size: int,
                                                    batch_size: int,
                                                    max_tokens_list: list[int],
        ) -> None:
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    sampling_method = 'greedy'
    
    logger.info(f"cache_type: {cache_type}, num_blocks: {num_blocks}, block_size: {block_size}")
    
    engine = InferenceEngine(model = model,
                             device = device,
                             tokenizer = tokenizer,
                             eos_token_id = model_cfg['eos_token_id'],
                             sampling_method = sampling_method,
                             is_kv_cache_enabled = True,
                             max_tokens_for_kv_cache = model_cfg['n_ctx'],
                             batch_size = batch_size,
                             model_cfg = model_cfg,
                             cache_type = cache_type,
                             num_blocks = num_blocks,
                             block_size = block_size,
                )
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = model.num_parameters(),
                               dtype = next(model.parameters()).dtype
                )
    
    
    # Warmup
    engine.generate(["Hello"] * batch_size, max_tokens = 2)
    
    prompt = ["Deep learning has revolutionized the field of",
                "The meaning of life is",
                "In a galaxy far far away",
                "The key to understanding transformers is"]
    
    results = []
    for max_tokens in tqdm(max_tokens_list):
        
        metrics.start()
        output = engine.generate(input_text = prompt, max_tokens = max_tokens)
        total_tokens = sum(r['token_count'] for r in output) if isinstance(output, list) else output['token_count']
        metrics.stop(num_tokens = total_tokens)
        
        result = metrics.result()
        result['max_tokens'] = max_tokens
        result['total_tokens'] = total_tokens
        results.append(result)
        
    # Print summary table
    header = (
        f"{'Max Tokens':<14} {'Total Tokens':<14} {'Latency (s)':<14} "
        f"{'Tok/s':<9} {'Peak Mem (MB)':<15} {'GPU Util (%)':<15} {'MFU (%)':<10}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        row = (
            f"{r['max_tokens']:<14} {r['total_tokens']:<14} "
            f"{r['latency_sec']:<14.3f} {r['tokens_per_sec']:<9.1f} "
            f"{r['peak_memory_mb']:<15.1f} "
            f"{r['gpu_snapshot']['gpu_utilization_percent']:<15.1f} "
            f"{r['MFU_percent']:<10.2f}"
        )
        logger.info(row)

def run_fragmentation_vs_block_size_benchmark(cache_type: str, 
                                              max_tokens: int,
                                              batch_size: int,
                                              total_capacity: int,
                                              block_sizes: list[int],
    ) -> None:
    # This benchmark would measure the fragmentation of the KV cache for different block sizes.
    # Fragmentation can be measured as the ratio of used memory to allocated memory in the KV cache.
    

    config, model_cfg = load_asset_paths()

    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    sampling_method = 'greedy'
    
    # # Warmup
    # engine.generate(["Hello"] * batch_size, max_tokens = 2)
    
    prompt = ["Deep learning has revolutionized the field of",
              "The meaning of life is",
              "In a galaxy far far away",
              "The key to understanding transformers is"]
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = model.num_parameters(),
                               dtype = next(model.parameters()).dtype
                )
    
    results = []
    for block_size in tqdm(block_sizes):
        
        num_blocks = total_capacity // block_size
        
        logger.info(f"cache_type: paged, block_size: {block_size}, num_blocks: {num_blocks}, total_capacity: {total_capacity}")

        engine = InferenceEngine(model = model,
                             device = device,
                             tokenizer = tokenizer,
                             eos_token_id = model_cfg['eos_token_id'],
                             sampling_method = sampling_method,
                             is_kv_cache_enabled = True,
                             max_tokens_for_kv_cache = model_cfg['n_ctx'],
                             batch_size = batch_size,
                             model_cfg = model_cfg,
                             cache_type = 'paged',
                             num_blocks = num_blocks,
                             block_size = block_size,
                )
        
        # Warmup
        engine.generate(["Hello"] * batch_size, max_tokens = 2)
    
        # Compute fragmentation
        blocks_per_seq = math.ceil(max_tokens / block_size)
        allocated_slots = blocks_per_seq * block_size
        wasted_slots = allocated_slots - max_tokens
        fragmentation_pct = (wasted_slots / allocated_slots) * 100

        metrics.start()
        output = engine.generate(input_text = prompt, max_tokens = max_tokens)
        total_tokens = sum(r['token_count'] for r in output) if isinstance(output, list) else output['token_count']
        metrics.stop(num_tokens = total_tokens)
        
        result = metrics.result()
        result['total_tokens'] = total_tokens
        result['block_size'] = block_size
        result['num_blocks'] = num_blocks
        result['blocks_per_seq'] = blocks_per_seq
        result['allocated_slots'] = allocated_slots
        result['wasted_slots'] = wasted_slots
        result['fragmentation_pct'] = fragmentation_pct
        results.append(result)
        
    # Print summary table
    header = (
        f"{'Block Size':<12} {'Num Blocks':<12} {'Blks/Seq':<10} "
        f"{'Alloc Slots':<13} {'Wasted':<8} {'Frag (%)':<10} "
        f"{'Tok/s':<9} {'Latency (s)':<13} {'Peak Mem (MB)':<15}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        row = (
            f"{r['block_size']:<12} {r['num_blocks']:<12} {r['blocks_per_seq']:<10} "
            f"{r['allocated_slots']:<13} {r['wasted_slots']:<8} {r['fragmentation_pct']:<10.1f} "
            f"{r['tokens_per_sec']:<9.1f} {r['latency_sec']:<13.3f} {r['peak_memory_mb']:<15.1f}"
        )
        logger.info(row)
   
def run_allocator_pressure_benchmark(batch_sizes: list[int],
                                     num_blocks: int,
                                     block_size: int,
                                     max_tokens: int,
    ) -> None:
    # This benchmark would measure the latency and GPU memory usage of the KV cache under different levels of allocator pressure.
    # Allocator pressure can be simulated by varying the batch size, which affects how many sequences are being processed in parallel and thus how many blocks are allocated in the KV cache.
    
    config, model_cfg = load_asset_paths()
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sampling_method = 'greedy'
    
    total_capacity = num_blocks * block_size
    blocks_per_seq = math.ceil(max_tokens / block_size)
    
    logger.info(f"Pool: {num_blocks} blocks × {block_size} = {total_capacity} slots | {blocks_per_seq} blocks/seq")
    
    with open('assets/files/prompt_1024.txt', 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]),
                               num_params = model.num_parameters(),
                               dtype = next(model.parameters()).dtype)
    
    results = []
    for bs in tqdm(batch_sizes):
        
        blocks_needed = bs * blocks_per_seq
        fits = "Yes" if blocks_needed <= num_blocks else "No"
        
        engine = InferenceEngine(model = model,
                                 device = device,
                                 tokenizer = tokenizer,
                                 eos_token_id = model_cfg['eos_token_id'],
                                 sampling_method = sampling_method,
                                 is_kv_cache_enabled = True,
                                 max_tokens_for_kv_cache = model_cfg['n_ctx'],
                                 batch_size = bs,
                                 model_cfg = model_cfg,
                                 cache_type = 'paged',
                                 num_blocks = num_blocks,
                                 block_size = block_size,
                    )
        
        batch_prompts = prompts[:bs]
        
        try:
            engine.generate(batch_prompts, max_tokens = 2)  # warmup
            
            metrics.start()
            output = engine.generate(batch_prompts, max_tokens)
            total_tokens = sum(r['token_count'] for r in output) if isinstance(output, list) else output['token_count']
            metrics.stop(num_tokens = total_tokens)
            
            result = metrics.result()
            result['batch_size'] = bs
            result['blocks_needed'] = blocks_needed
            result['fits'] = fits
            result['utilization_pct'] = (blocks_needed / num_blocks) * 100
            result['total_tokens'] = total_tokens
            result['status'] = 'OK'
            results.append(result)
            
        except Exception as e:
            results.append({
                'batch_size': bs,
                'blocks_needed': blocks_needed,
                'fits': fits,
                'utilization_pct': (blocks_needed / num_blocks) * 100,
                'status': f'FAIL: {type(e).__name__}',
                'total_tokens': 0,
                'tokens_per_sec': 0,
                'latency_sec': 0,
                'peak_memory_mb': 0,
            })
    
    # Print summary table
    header = (
        f"{'Batch':<7} {'Blks Needed':<13} {'Fits?':<7} {'Util (%)':<10} "
        f"{'Status':<25} {'Tok/s':<9} {'Latency (s)':<13} {'Peak Mem (MB)':<15}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        row = (
            f"{r['batch_size']:<7} {r['blocks_needed']:<13} {r['fits']:<7} {r['utilization_pct']:<10.1f} "
            f"{r['status']:<25} {r['tokens_per_sec']:<9.1f} {r['latency_sec']:<13.3f} {r['peak_memory_mb']:<15.1f}"
        )
        logger.info(row)
        
if __name__ == "__main__":
    
    block_size = 16
    for num_blocks in [64, 128, 256]:
        run_throughput_peak_gpu_vs_block_benchmark(cache_type = 'standard', 
                                                   num_blocks = num_blocks, 
                                                   block_size = block_size)
        run_throughput_peak_gpu_vs_block_benchmark(cache_type = 'paged', 
                                                   num_blocks = num_blocks, 
                                                   block_size = block_size)
        
    block_size = 16
    num_blocks = 1900
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    run_throughput_peak_gpu_vs_batch_size_benchmark(cache_type = 'standard', 
                                                    num_blocks = num_blocks, 
                                                    block_size = block_size,
                                                    batch_sizes = batch_sizes)
    run_throughput_peak_gpu_vs_batch_size_benchmark(cache_type = 'paged', 
                                                    num_blocks = num_blocks, 
                                                    block_size = block_size,
                                                    batch_sizes = batch_sizes)
    
    max_tokens_list = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    num_blocks = 256
    block_size = 16
    batch_size = 4
    
    run_throughput_peak_gpu_vs_max_tokens_benchmark(cache_type = 'standard',
                                                    num_blocks = num_blocks,
                                                    block_size = block_size,
                                                    batch_size = batch_size,
                                                    max_tokens_list = max_tokens_list)
    run_throughput_peak_gpu_vs_max_tokens_benchmark(cache_type = 'paged',
                                                    num_blocks = num_blocks,
                                                    block_size = block_size,
                                                    batch_size = batch_size,
                                                    max_tokens_list = max_tokens_list)
    
    block_sizes = [4, 8, 16, 32, 64]
    batch_size = 4
    total_capacity = 4096
    max_tokens = 50
    
    run_fragmentation_vs_block_size_benchmark(cache_type = 'paged',
                                              max_tokens = max_tokens,
                                              batch_size = batch_size,
                                              total_capacity = total_capacity,
                                              block_sizes = block_sizes)
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    run_allocator_pressure_benchmark(batch_sizes = batch_sizes,
                                     num_blocks = 64,
                                     block_size = 16,
                                     max_tokens = 50)
    