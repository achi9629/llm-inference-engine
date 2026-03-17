import torch, logging # type: ignore

from llm_engine import load_model, load_asset_paths, Tokenizer, InferenceEngine
from benchmarks.metrics import BenchmarkMetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

def run_throughput_benchmark():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    max_tokens = 50
    sampling_method = 'greedy'
    
    engine = InferenceEngine(model = model, 
                    device = device, 
                    tokenizer = tokenizer,
                    eos_token_id = model_cfg['eos_token_id'],
                    sampling_method = sampling_method
        )
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = model.num_parameters(),
                               dtype = next(model.parameters()).dtype)
    
    # Warmup (not timed)
    engine.generate("warmup")
    
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
        
if __name__ == "__main__":
    run_throughput_benchmark()