import uvicorn, argparse, torch

from llm_engine import load_asset_paths, load_model, create_app, load_scheduler_config, \
                       load_server_config
from llm_engine import Tokenizer, InferenceEngine, ContinuousBatchingScheduler, \
                       RequestHandler, Router

def start_server(args: argparse = None) -> None:
    
    config, model_cfg = load_asset_paths()
    scheduler_cfg = load_scheduler_config()
    server_cfg = load_server_config()
    
    model = load_model(config, model_cfg)
    
    tokenizer = Tokenizer(config)
    
    device = "cuda:" + str(args.device) if torch.cuda.is_available() else "cpu"
    
    engine = InferenceEngine(model = model,
                             device = device,
                             tokenizer = tokenizer,
                             eos_token_id = model_cfg['eos_token_id'],
                             sampling_method = 'greedy',
                             is_kv_cache_enabled = True,
                             max_tokens_for_kv_cache = model_cfg['n_ctx'],
                             batch_size = 1,
                             model_cfg = model_cfg,
                             cache_type = args.cache_type,
                             num_blocks = 512,
                             block_size = 16)
    
    scheduler = ContinuousBatchingScheduler(max_batch_size = scheduler_cfg['max_batch_size'])
    
    handler = RequestHandler(tokenizer = tokenizer, max_model_len = model_cfg['n_ctx'])
    
    router = Router(request_handler = handler, 
                    scheduler = scheduler, 
                    engine = engine,
                    async_mode = args.async_mode)
    
    app = create_app(router = router,
                     max_concurrent_requests = server_cfg['max_concurrent_requests'],
                     request_timeout = server_cfg['timeout'])
    
    uvicorn.run(app, host = server_cfg['host'], port = server_cfg['port']
        )
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run the LLM inference server")
    parser.add_argument("--device", type=int, default = 0, help="Device to run the model on (e.g., 0,1,2,3 for CUDA devices)")
    parser.add_argument("--cache_type", type=str, default = 'paged', help="Type of KV cache to use (e.g., 'paged', 'full')")
    parser.add_argument("--async_mode", action='store_true', help="Whether to use asynchronous processing for requests")
    
    return parser.parse_args()
    
if __name__ == "__main__":
    
    args = parse_args()
    start_server(args)
    
    
                                            
    