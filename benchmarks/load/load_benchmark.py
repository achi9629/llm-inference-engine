import torch, logging, json, argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Any

from benchmarks.metrics import BenchmarkMetrics
from llm_engine import load_asset_paths, load_model, Tokenizer, InferenceEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

def build_batch_prompts(prompts: List[str], bs: int) -> List[str]:
    if bs <= len(prompts):
        return prompts[:bs]
    repeats = (bs + len(prompts) - 1) // len(prompts)
    return (prompts * repeats)[:bs]

def run_oom_stress_test(cache_type: str,
                        batch_sizes: List[int],
                        num_blocks: int,
                        block_size: int,
                        max_tokens: int,
                        out_path: str | None = None,
        ) -> None:
    
    config, model_cfg = load_asset_paths()
    model = load_model(config, model_cfg)
    tokenizer = Tokenizer(config)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    with open('assets/files/prompt_1024.txt', 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
    if not prompts:
        raise ValueError("Prompt file is empty.")
    
    metrics = BenchmarkMetrics(device_id = int(device[-1]), 
                               num_params = model.num_parameters(),
                               dtype = next(model.parameters()).dtype
                )
    
    logger.info(f"Running cache_type={cache_type}, num_blocks={num_blocks}, block_size={block_size}")
    results: List[dict[str, Any]] = []
    
    for bs in tqdm(batch_sizes, desc=f"{cache_type}"):
        
        batch_prompts = build_batch_prompts(prompts, bs)
        engine = None
        
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        try:
            engine = InferenceEngine(model = model,
                                    device = device,
                                    tokenizer = tokenizer,
                                    eos_token_id = model_cfg["eos_token_id"],
                                    sampling_method = "greedy",
                                    is_kv_cache_enabled = True,
                                    max_tokens_for_kv_cache = model_cfg["n_ctx"],
                                    batch_size = bs,
                                    model_cfg  = model_cfg,
                                    cache_type = cache_type,
                                    num_blocks = num_blocks,
                                    block_size = block_size,)
            
            if bs == 1:
                # warmup run to exclude any one-time setup costs from metrics
                 engine.generate(batch_prompts, max_tokens=max_tokens)
            
            metrics.start()
            output = engine.generate(batch_prompts, max_tokens=max_tokens)
            total_tokens = sum(r["token_count"] for r in output) if isinstance(output, list) else output["token_count"]
            metrics.stop(num_tokens=total_tokens)
            
            result = metrics.result()
            result["status"] = "OK"
            result["batch_size"] = bs
            results.append(result)
            logger.info(
                f"Batch Size: {bs}, Cache Type: {cache_type}, "
                f"Peak Mem: {result['peak_memory_mb']:.1f} MB, Status: OK"
            )
        
        except (RuntimeError, MemoryError) as e:
            msg = str(e).lower()
            
            if "out of memory" in msg or "not enough free blocks" in msg:
                status = "OOM" if "out of memory" in msg else "Not enough free blocks"
                logger.warning(f"{status} at batch size {bs} for cache type {cache_type}")

                results.append({"status": status,
                                "batch_size": bs,
                                "error": str(e),
                                "num_tokens": None,
                                "latency_sec": None,
                                "tokens_per_sec": None,
                                "peak_memory_mb": None,
                                })
                break
            raise
        finally:
            if engine is not None:
                del engine
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            
    if out_path:
        payload = {
            "cache_type": cache_type,
            "num_blocks": num_blocks,
            "block_size": block_size,
            "max_tokens": max_tokens,
            "batch_sizes": batch_sizes,
            "results": results,
        }
        Path(out_path).write_text(json.dumps(payload, indent=2))
        logger.info(f"Saved results to {out_path}")
    
    return results

def compare_results(standard_json: str, 
                    paged_json: str, 
                    batch_sizes: List[int]
    ) -> None:
    
    std_data = json.loads(Path(standard_json).read_text())
    paged_data = json.loads(Path(paged_json).read_text())
    
    
    std_results = {r["batch_size"]: r for r in std_data["results"]}
    paged_results = {r["batch_size"]: r for r in paged_data["results"]}

    header = (
        f"{'Batch Size':<12} {'Std Mem (MB)':<15} {'Std Status':<20} "
        f"{'Paged Mem (MB)':<18} {'Paged Status':<22} {'Ratio (Std/Paged)':<17}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    
    for bs in batch_sizes:
        std = std_results.get(bs)
        paged = paged_results.get(bs)

        std_mem = f"{std['peak_memory_mb']:<15.1f}" if std and std.get("peak_memory_mb") is not None else f"{'N/A':<15}"
        std_status = std["status"] if std else "-"

        paged_mem = f"{paged['peak_memory_mb']:<18.1f}" if paged and paged.get("peak_memory_mb") is not None else f"{'N/A':<18}"
        paged_status = paged["status"] if paged else "-"

        ratio = "N/A"
        if std and paged and std.get("peak_memory_mb") and paged.get("peak_memory_mb"):
            ratio = f"{(std['peak_memory_mb'] / paged['peak_memory_mb']):<17.1f}"

        logger.info(f"{bs:<12} {std_mem}{std_status:<20} {paged_mem}{paged_status:<22} {ratio}")
        
    
def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="OOM stress benchmark for standard vs paged KV cache")
    parser.add_argument("--mode", choices=["run", "compare"], default="run")
    parser.add_argument("--cache_type", choices=["standard", "paged"], default="standard")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--standard_json", type=str, default=None)
    parser.add_argument("--paged_json", type=str, default=None)
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    num_blocks = 4096
    block_size = 16
    max_tokens = 50
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    if args.mode == "run":
        run_oom_stress_test(cache_type = args.cache_type,
                            batch_sizes = batch_sizes,
                            num_blocks = num_blocks,
                            block_size = block_size,
                            max_tokens = max_tokens,
                            out_path = args.out,
            )   
    else:
        if not args.standard_json or not args.paged_json:
            raise ValueError("For compare mode, provide --standard_json and --paged_json")
        
        compare_results(standard_json = args.standard_json,
                        paged_json = args.paged_json,
                        batch_sizes= batch_sizes,)