import uvicorn
from llm_engine import load_asset_paths, load_model, create_app
from llm_engine import Tokenizer, InferenceEngine, ContinuousBatchingScheduler, \
                       RequestHandler, Router

def start_server():
    
    config, model_cfg = load_asset_paths()
    
    model = load_model(config, model_cfg)
    
    tokenizer = Tokenizer(config)
    
    device = 'cuda:0'
    batch_size = 1
    max_batch_size = 4
    
    engine = InferenceEngine(model = model,
                             device = device,
                             tokenizer = tokenizer,
                             eos_token_id = model_cfg['eos_token_id'],
                             sampling_method = 'greedy',
                             is_kv_cache_enabled = True,
                             max_tokens_for_kv_cache = model_cfg['n_ctx'],
                             batch_size = batch_size,
                             model_cfg = model_cfg,
                             cache_type = 'paged',
                             num_blocks = 128,
                             block_size = 16)
    
    scheduler = ContinuousBatchingScheduler(max_batch_size = max_batch_size)
    
    handler = RequestHandler(tokenizer = tokenizer, max_model_len = model_cfg['n_ctx'])
    
    router = Router(request_handler = handler, scheduler = scheduler, engine = engine)
    
    app = create_app(router)
    
    uvicorn.run(app, host = "127.0.0.1", port = 8000)
    
if __name__ == "__main__":
    
    start_server()
    
    
                                            
    