import os
import json
import torch
import logging
# from time import perf_counter
from vllm import LLM, SamplingParams

from benchmarks.metrics import BenchmarkMetrics
from src.llm_engine import load_model, load_asset_paths, Tokenizer, InferenceEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

def run_latency_benchmark(is_kv_cache_enabled: bool = False) -> None:
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    
    max_tokens = 50
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
    engine.generate("warmup")
    
    prompt_length = [64, 256, 512]
    results = []
    
    for length in prompt_length:
        
        token_ids = torch.randint(0, model_cfg['vocab_size'], (length,))
        prompt = tokenizer.decode(token_ids)
        metrics.start()
        output = engine.generate(input_text = prompt, max_tokens = max_tokens)
        metrics.stop(num_tokens = output['token_count'])
        
        result = metrics.result()
        result['prompt_length'] = length
        results.append(result)
        
        logger.info(f"Prompt length: {length} tokens")
        metrics.summary()
        logger.info("")
        
    # Summary table
    header = (
        f"{'Prompt Len':<12} {'Latency (s)':<14} {'Tok/s':<10} "
        f"{'Peak Mem (MB)':<15} {'GPU Util (%)':<14} {'Mem Util (%)':<14} {'MFU (%)':<10}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        row = (
            f"{r['prompt_length']:<12} {r['latency_sec']:<14.3f} {r['tokens_per_sec']:<10.1f} "
            f"{r['peak_memory_mb']:<15.1f} {r['gpu_snapshot']['gpu_utilization_percent']:<14.1f} "
            f"{r['gpu_snapshot']['memory_utilization_percent']:<14.1f} {r['MFU_percent']:<10.2f}"
        )
        logger.info(row)
        
    # Save results to JSON
    output_dir = "outputs/results/gpt2/latency"
    os.makedirs(output_dir, exist_ok=True)
    tag = "kv_cache" if is_kv_cache_enabled else "no_cache"
    output_path = os.path.join(output_dir, f"latency_{tag}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    
def run_vllm_latency_benchmark() -> None:

    # model_name = "/pfs01/performance-tier/rd_ici/algo_train/acsingh/projects/finetune/assets/models/Mistral-7B-v0.3/"
    model_name = '/pfs01/performance-tier/rd_ici/algo_train/acsingh/projects/engine/assets/models/gpt2_hf/'
    device_id = 0
    max_tokens = 50
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    llm = LLM(model=model_name, dtype="float32", gpu_memory_utilization=0.1)
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos = True)

    # Get model param count for MFU
    num_params = sum(p.numel() for p in llm.llm_engine.model_executor.driver_worker.model_runner.model.parameters())
    dtype = torch.float32
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = num_params,
                               dtype = dtype)

    # Warmup
    llm.generate(["warmup"], sampling_params)

    prompt_lengths = [64, 256, 512]
    tokenizer = llm.get_tokenizer()
    results = []

    for length in prompt_lengths:
        # Generate random prompt of target length
        token_ids = torch.randint(0, tokenizer.vocab_size, (length,)).tolist()
        prompt = tokenizer.decode(token_ids)

        metrics.start()
        outputs = llm.generate([prompt], sampling_params)
        num_generated = len(outputs[0].outputs[0].token_ids)
        metrics.stop(num_tokens=num_generated)
        
        result = metrics.result()
        result['prompt_length'] = length
        results.append(result)
        
        logger.info(f"Prompt length: {length} tokens")
        metrics.summary()
        logger.info("")
        
    # Summary table
    header = (
        f"{'Prompt Len':<12} {'Latency (s)':<14} {'Tok/s':<10} "
        f"{'Peak Mem (MB)':<15} {'GPU Util (%)':<14} {'Mem Util (%)':<14} {'MFU (%)':<10}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        row = (
            f"{r['prompt_length']:<12} {r['latency_sec']:<14.3f} {r['tokens_per_sec']:<10.1f} "
            f"{r['peak_memory_mb']:<15.1f} {r['gpu_snapshot']['gpu_utilization_percent']:<14.1f} "
            f"{r['gpu_snapshot']['memory_utilization_percent']:<14.1f} {r['MFU_percent']:<10.2f}"
        )
        logger.info(row)
    # Save results
    output_dir = "outputs/results/gpt2/latency"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "latency_vllm.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    
if __name__ == "__main__":
    # run_latency_benchmark()
    # run_latency_benchmark(is_kv_cache_enabled = True)
    run_vllm_latency_benchmark()