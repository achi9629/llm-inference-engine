from .GPT2.transformer import GPT2
from llm_engine.utils.weight_loader import load_weights


def load_model(config: dict, 
               model_cfg: dict, 
               is_weights: bool = True
    ) -> object:
    
    '''
    Load the model based on the provided configuration.
    '''
    
    if config['model_family'] == "gpt2" and config['model_variant'] == '124M':
        model = GPT2(model_cfg)
    else:
        raise NotImplementedError(f"Model family {config['model_family']} with variant {config['model_variant']} is not implemented.")
        
    if is_weights:
        model = load_weights(model, config)
    return model