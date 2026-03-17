import torch, logging # type: ignore

from llm_engine import load_model, load_asset_paths, Tokenizer, InferenceEngine, InferenceProfiler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

def run_profiler_benchmark():
    
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
    
    # Warmup (not timed)
    engine.generate("warmup")
    
    profiler = InferenceProfiler(use_cuda = True, 
                                 record_shapes = True,
                                 with_stack = True,
    )
    
    with profiler.profile():
        engine.generate("The capital of France is", max_tokens = max_tokens)
        
    logger.info(profiler.summary(top_n = 15, sort_by = 'cuda_time_total'))
    logger.info(profiler.summary(top_n = 15, sort_by = 'self_cuda_time_total'))
    
    # Export the trace to a JSON file for visualization in Chrome DevTools or other profiling tools
    # profiler.export_trace("profiler_trace.json")
    
if __name__ == "__main__":
    run_profiler_benchmark()