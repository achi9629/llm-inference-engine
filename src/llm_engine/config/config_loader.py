from pathlib import Path
import json
import yaml

def load_asset_paths(model_cfg_path: str = "configs/model_config.yaml") -> dict[str, str]:
    
    '''
    Description:
    '''
    
    repo_root = Path(__file__).resolve().parents[3]  # adjust if file location differs
    
    # 1) read model_config.yaml
    with open(repo_root / model_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) 
        
    model_dir = repo_root / cfg["model_dir"]
    
    # 2) read manifest.json inside selected model dir
    with open(model_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # 3) build paths
    config = {
        "model_family": manifest["model_family"],
        "model_variant": manifest["model_variant"],
        "architecture": manifest["architecture"],
        "model_dir": str(model_dir),
        "weights": str(model_dir / manifest["weights_file"]),
        "config": str(model_dir / manifest["config_file"]),
        "tokenizer_type": manifest["tokenizer"]["type"],
        "vocab": manifest["tokenizer"]["vocab_file"],
        "merges": manifest["tokenizer"]["merges_file"]
    }
    
    model_cfg = json.load(open(config["config"], "r", encoding="utf-8"))
    
    return config, model_cfg

def load_scheduler_config(path: str = "configs/scheduler_config.yaml") -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    with open(repo_root / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_server_config(path: str = "configs/server_config.yaml") -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    with open(repo_root / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

