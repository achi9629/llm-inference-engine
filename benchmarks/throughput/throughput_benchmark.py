import os
import json
import torch
import logging
from vllm import LLM, SamplingParams

from llm_engine import load_model, load_asset_paths, Tokenizer, InferenceEngine
from benchmarks.metrics import BenchmarkMetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

def run_throughput_benchmark(is_kv_cache_enabled: bool = False) -> None:
    
    '''
    Description:
        This benchmark evaluates the throughput of a language model by measuring the time taken to generate a
        specified number of tokens given a fixed prompt. For different max token lengths, it records the latency,
        throughput, and GPU metrics such as memory usage and utilization. The benchmark can be run with or 
        without KV caching to compare the performance impact of caching on generation speed and resource usage.
    Args:
        is_kv_cache_enabled (bool): Whether to enable KV caching during generation. Defaults to False.
    Returns:
        None: The function prints the results to the console and does not return any value.
    '''
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    sampling_method = 'greedy'
    
    logger.info(f"KV Cache Enabled: {is_kv_cache_enabled}")
    
    engine = InferenceEngine(model = model, 
                    device = device, 
                    tokenizer = tokenizer,
                    eos_token_id = model_cfg['eos_token_id'],
                    sampling_method = sampling_method,
                    is_kv_cache_enabled = is_kv_cache_enabled,
                    max_tokens_for_kv_cache = model_cfg['n_ctx'],
                    batch_size = 1,
                    model_cfg = model_cfg
        )
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = model.num_parameters(),
                               dtype = next(model.parameters()).dtype)
    
    # Warmup (not timed)
    engine.generate("warmup", max_tokens = 10)
    
    prompt = "The meaning of life is" 
    max_tokens = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    results = []
    
    for tokens in max_tokens:
        
        metrics.start()
        output = engine.generate(input_text = prompt, max_tokens = tokens)
        metrics.stop(num_tokens = output['token_count'])
        
        result = metrics.result()
        result['Max_Tokens'] = tokens
        result['Actual_Tokens'] = output['token_count']
        results.append(result)
        
        logger.info(f"Max Tokens Length: {tokens} tokens")
        metrics.summary()
        logger.info("")
        
    # Summary table
    header = (
        f"{'Max Tokens':<12} {'Actual Tokens':<14} {'Latency (s)':<14} {'Tok/s':<9} "
        f"{'Peak Mem (MB)':<15} {'GPU Util (%)':<15} {'Mem Util (%)':<15} {'MFU (%)':<10}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        row = (
            f"{r['Max_Tokens']:<12} {r['Actual_Tokens']:<14} {r['latency_sec']:<14.3f} {r['tokens_per_sec']:<9.1f} "
            f"{r['peak_memory_mb']:<15.1f} {r['gpu_snapshot']['gpu_utilization_percent']:<15.1f} "
            f"{r['gpu_snapshot']['memory_utilization_percent']:<15.1f} {r['MFU_percent']:<10.2f}"
        )
        logger.info(row)
        
    output_dir = "outputs/results/gpt2/benchmark"
    os.makedirs(output_dir, exist_ok=True)
    tag = "kv_cache" if is_kv_cache_enabled else "no_cache"
    output_path = os.path.join(output_dir, f"throughput_{tag}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
        
def run_vllm_throughput_benchmark() -> None:
    
    model_name = '/pfs01/performance-tier/rd_ici/algo_train/acsingh/projects/engine/assets/models/gpt2_hf/'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    llm = LLM(model=model_name, dtype="float32", gpu_memory_utilization=0.1)

    num_params = sum(p.numel() for p in llm.llm_engine.model_executor.driver_worker.model_runner.model.parameters())
    dtype = torch.float32
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = num_params,
                               dtype = dtype)
    
    # Warmup
    llm.generate(["warmup"], SamplingParams(temperature=0, max_tokens=10, ignore_eos=True))
    
    prompt = "The meaning of life is"
    max_tokens = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    results = []
    
    for tokens in max_tokens:
        
        sampling_params = SamplingParams(temperature=0, max_tokens=tokens, ignore_eos=True)
        
        metrics.start()
        outputs = llm.generate([prompt], sampling_params)
        num_generated = len(outputs[0].outputs[0].token_ids)
        metrics.stop(num_tokens=num_generated)
        
        result = metrics.result()
        result['Max_Tokens'] = tokens
        result['Actual_Tokens'] = num_generated
        results.append(result)
        
        logger.info(f"Max Tokens Length: {tokens} tokens")
        metrics.summary()
        logger.info("")
        
    # Summary table
    header = (
        f"{'Max Tokens':<12} {'Actual Tokens':<14} {'Latency (s)':<14} {'Tok/s':<9} "
        f"{'Peak Mem (MB)':<15} {'GPU Util (%)':<15} {'Mem Util (%)':<15} {'MFU (%)':<10}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        row = (
            f"{r['Max_Tokens']:<12} {r['Actual_Tokens']:<14} {r['latency_sec']:<14.3f} {r['tokens_per_sec']:<9.1f} "
            f"{r['peak_memory_mb']:<15.1f} {r['gpu_snapshot']['gpu_utilization_percent']:<15.1f} "
            f"{r['gpu_snapshot']['memory_utilization_percent']:<15.1f} {r['MFU_percent']:<10.2f}"
        )
        logger.info(row)
        
    output_dir = "outputs/results/gpt2/benchmark"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "throughput_vllm.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

def run_batch_throughput_benchmark(is_kv_cache_enabled: bool = True) -> None:
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    max_tokens = 50
    sampling_method = 'greedy'
    
    logger.info(f"KV Cache Enabled: {is_kv_cache_enabled}")
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = model.num_parameters(),
                               dtype = next(model.parameters()).dtype)
    
    with open('assets/files/prompt_1024.txt', 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    results = []
    
    for bs in batch_sizes:
    
        engine = InferenceEngine(model = model, 
                        device = device, 
                        tokenizer = tokenizer,
                        eos_token_id = model_cfg['eos_token_id'],
                        sampling_method = sampling_method,
                        is_kv_cache_enabled = is_kv_cache_enabled,
                        max_tokens_for_kv_cache = model_cfg['n_ctx'],
                        batch_size = bs,
                        model_cfg = model_cfg
            )
        
        batch_prompts = prompts[:bs]
        
        # Warmup
        engine.generate(batch_prompts, max_tokens)
        
        # Timed run
        metrics.start()
        output = engine.generate(batch_prompts, max_tokens)
        total_tokens = sum(r['token_count'] for r in output) if isinstance(output, list) else output['token_count']
        metrics.stop(num_tokens=total_tokens)
        
        result = metrics.result()
        result['batch_size'] = bs
        result['total_tokens'] = total_tokens
        results.append(result)
        
        logger.info(f"Batch Size: {bs}, Total Tokens: {total_tokens}")
        metrics.summary()
        logger.info("")
        
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
        
    output_dir = "outputs/results/gpt2/benchmark"
    os.makedirs(output_dir, exist_ok=True)
    tag = "kv_cache" if is_kv_cache_enabled else "no_cache"
    output_path = os.path.join(output_dir, f"batch_throughput_{tag}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
        
def run_vllm_batch_throughput_benchmark() -> None:

    model_name = '/pfs01/performance-tier/rd_ici/algo_train/acsingh/projects/engine/assets/models/gpt2_hf/'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    llm = LLM(model=model_name, dtype="float32", gpu_memory_utilization=0.9)

    num_params = sum(p.numel() for p in llm.llm_engine.model_executor.driver_worker.model_runner.model.parameters())
    dtype = torch.float32

    metrics = BenchmarkMetrics(device_id=int(device[-1]),
                               num_params=num_params,
                               dtype=dtype)

    with open('assets/files/prompt_1024.txt', 'r') as f:
        prompts = [line.strip() for line in f.readlines()]

    max_tokens = 50
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    results = []

    # Warmup
    llm.generate(["warmup"], SamplingParams(temperature=0, max_tokens=10, ignore_eos=True))

    for bs in batch_sizes:

        batch_prompts = prompts[:bs]
        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)

        metrics.start()
        outputs = llm.generate(batch_prompts, sampling_params)
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        metrics.stop(num_tokens=total_tokens)

        result = metrics.result()
        result['batch_size'] = bs
        result['total_tokens'] = total_tokens
        results.append(result)

        logger.info(f"Batch Size: {bs}, Total Tokens: {total_tokens}")
        metrics.summary()
        logger.info("")

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
        
    output_dir = "outputs/results/gpt2/benchmark"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "batch_throughput_vllm.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
        
if __name__ == "__main__":
    
    run_vllm_throughput_benchmark()
    run_throughput_benchmark()
    run_throughput_benchmark(is_kv_cache_enabled = True)
    run_batch_throughput_benchmark()
    run_vllm_batch_throughput_benchmark()