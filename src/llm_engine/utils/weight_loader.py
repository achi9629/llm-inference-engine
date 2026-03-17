import torch # type: ignore
import torch.nn as nn # type: ignore

import logging
logger = logging.getLogger(__name__)

def load_weights(model: nn.Module, config: dict) -> nn.Module:
    
    '''
    Loads weights into the given model based on the provided configuration.
    Check for missing, unexpected, and shape-mismatched keys, and logs the results.
    '''
    
    weights_path = config['weights']
    
    model_weights = torch.load(weights_path, weights_only=True, map_location='cpu')
    result = model.load_state_dict(model_weights, strict=False)
    
    # Determine which keys were loaded, missing, and unexpected
    loaded_keys = set(model_weights.keys()) - set(result.unexpected_keys)
    missing_keys = result.missing_keys
    unexpected_keys = result.unexpected_keys
    
    log_key_level_check(model, config, weights_path, loaded_keys, missing_keys, unexpected_keys)
    
    # Check for shape mismatches between the loaded weights and the model's expected shapes
    shape_mismatches = []
    for key in loaded_keys:
        ckpt_shape = model_weights[key].shape
        model_shape = model.state_dict()[key].shape
        if ckpt_shape != model_shape:
            shape_mismatches.append((key, ckpt_shape, model_shape))
    
    log_shape_mismatches(shape_mismatches, loaded_keys, model_weights)
    
    return model

def log_shape_mismatches(shape_mismatches: list, 
                         loaded_keys: set, 
                         model_weights: dict
                    ) -> None:
    
    '''
    Logs any shape mismatches between the loaded checkpoint weights and the model's expected shapes.
    '''
    
    logger.info(f"Shape matches: {len(loaded_keys) - len(shape_mismatches)} / {len(loaded_keys)}")

    if shape_mismatches:
        for key, ckpt_shape, model_shape in shape_mismatches:
            logger.error(f"Shape mismatch: {key} — checkpoint: {ckpt_shape}, model: {model_shape}")
            
    for key in sorted(loaded_keys):
        logger.debug(f"  {key}: {model_weights[key].shape}")

def log_key_level_check(model: nn.Module, 
                        config: dict, 
                        weights_path: str, 
                        loaded_keys: set, 
                        missing_keys: list, 
                        unexpected_keys: list
                    ) -> None:
    
    '''
    Logs a summary of the weight loading process, including the number of parameters loaded, 
    missing, and unexpected.
    '''
    
    logger.info(f"{'='*50}")
    logger.info(f"Model Family: {config['model_family']} and Variant: {config['model_variant']}")
    logger.info(f"Loaded weights from: {weights_path}")
    logger.info(f"Weight Loading Summary")
    logger.info(f"{'='*50}")
    logger.info(f"Loaded:     {len(loaded_keys)} / {len(model.state_dict())} parameters")
    logger.info(f"Missing:    {len(missing_keys)}")
    logger.info(f"Unexpected: {len(unexpected_keys)}")
    
    if missing_keys:
        logger.warning(f"\nMissing keys:")
        for k in missing_keys:
            logger.warning(f"  - {k}")

    if unexpected_keys:
        logger.warning(f"\nUnexpected keys:")
        for k in unexpected_keys:
            logger.warning(f"  - {k}")

    logger.info(f"{'='*50}\n")