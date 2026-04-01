import torch, statistics, logging
from typing import List, Tuple

from llm_engine import load_model, load_asset_paths
from llm_engine import Tokenizer, InferenceEngine
from benchmarks.metrics import BenchmarkMetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

WARMUP_ITERS = 3
MEASURED_ITERS = 17

def test_single_user_latency(prompt_length: int,
                             cache_type: str,
                             ) -> Tuple[dict, dict]:
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    
    tokenizer = Tokenizer(config)
    
    device = 'cuda:0'
    batch_size = 1
    max_tokens = 50
    
    engine = InferenceEngine(model = model,
                             device = device,
                             tokenizer = tokenizer,
                             eos_token_id = model_cfg['eos_token_id'],
                             sampling_method = 'greedy',
                             is_kv_cache_enabled = True,
                             max_tokens_for_kv_cache = model_cfg['n_ctx'],
                             batch_size = batch_size,
                             model_cfg = model_cfg,
                             cache_type = cache_type,
                             num_blocks = 128,
                             block_size = 16)
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = model.num_parameters(),
                               dtype = next(model.parameters()).dtype)
    
    token_ids = torch.randint(0, model_cfg['vocab_size'], (prompt_length,))
    prompt = tokenizer.decode(token_ids)
    
    # --- Cold: first generate() on a fresh engine ---
    metrics.start()
    output = engine.generate(input_text = prompt, max_tokens = max_tokens)
    metrics.stop(num_tokens = output['token_count'])
    cold_result = metrics.result()
    cold_result['prompt_length'] = prompt_length
    cold_result['cache_type'] = cache_type
    cold_result['phase'] = 'Cold'
    
    logger.info("Prompt Length: %d, Cache Type: %s, Phase: Cold", prompt_length, cache_type)
    metrics.summary()
    
    # --- Warmup: discard first few iterations ---
    for _ in range(WARMUP_ITERS):
        engine.generate(input_text = prompt, max_tokens = max_tokens)
    
    # --- Warm: measured iterations ---
    latency_track, peak_memory_track, tokens_per_sec_track = [], [], []
    for _ in range(MEASURED_ITERS):
        metrics.start()
        output = engine.generate(input_text = prompt, max_tokens = max_tokens)
        metrics.stop(num_tokens = output['token_count'])
        
        result = metrics.result()
        latency_track.append(result['latency_sec'])
        peak_memory_track.append(result['peak_memory_mb'])
        tokens_per_sec_track.append(result['tokens_per_sec'])
    
    warm_result = result.copy()
    warm_result['latency_sec_mean'] = statistics.mean(latency_track)
    warm_result['latency_sec_std'] = statistics.stdev(latency_track)
    warm_result['peak_memory_mb_mean'] = statistics.mean(peak_memory_track)
    warm_result['peak_memory_mb_std'] = statistics.stdev(peak_memory_track)
    warm_result['tokens_per_sec_mean'] = statistics.mean(tokens_per_sec_track)
    warm_result['tokens_per_sec_std'] = statistics.stdev(tokens_per_sec_track)
    warm_result['prompt_length'] = prompt_length
    warm_result['cache_type'] = cache_type
    warm_result['phase'] = 'Warm'
    
    logger.info("Prompt Length: %d, Cache Type: %s, Phase: Warm", prompt_length, cache_type)
    metrics.summary()
    
    return cold_result, warm_result

def display_results(results: List[dict]) -> None:
    
    header = (
        "%-10s %-12s %-12s %-18s %-18s %-18s"
        % ("Phase", "Cache", "Prompt Len", "Latency (s)", "Tok/s", "Peak Mem (MB)")
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        if r['phase'] == 'Cold':
            latency = "%.3f" % r['latency_sec']
            token = "%.1f" % r['tokens_per_sec']
            memory = "%.1f" % r['peak_memory_mb']
        else:
            latency = "%.3f\u00b1%.3f" % (r['latency_sec_mean'], r['latency_sec_std'])
            token = "%.1f\u00b1%.1f" % (r['tokens_per_sec_mean'], r['tokens_per_sec_std'])
            memory = "%.1f\u00b1%.1f" % (r['peak_memory_mb_mean'], r['peak_memory_mb_std'])
        row = "%-10s %-12s %-12s %-18s %-18s %-18s" % (
            r['phase'], r['cache_type'], r['prompt_length'], latency, token, memory
        )
        logger.info(row)
    
if __name__ == "__main__":
    
    prompt_lengths = [32, 256, 512]
    cache_types = ['standard', 'paged']
    results = []
    for prompt_len in prompt_lengths:
        for cache in cache_types:
            cold, warm = test_single_user_latency(prompt_len, cache)
            results.append(cold)
            results.append(warm)
    display_results(results)
