import asyncio, time, statistics, logging, argparse, json, httpx
import numpy as np
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROMPT = {
    "short": "The capital of France is",
    "medium": "Deep learning has revolutionized the field of artificial intelligence by enabling models to learn complex patterns from large datasets without explicit feature engineering",
    "long": "The theory of relativity, proposed by Albert Einstein in the early twentieth century, fundamentally changed our understanding of space, time, and gravity. "
            "It consists of two interrelated physics theories: special relativity and general relativity. Special relativity applies to all physical phenomena in the absence of gravity, "
            "while general relativity explains the law of gravitation and its relation to the forces of nature. The theory transformed theoretical physics and astronomy during the twentieth century, "
            "superseding a two hundred year old theory of mechanics created primarily by Isaac Newton. It introduced concepts including four dimensional spacetime as a unified entity of space and time, "
            "relativity of simultaneity, kinematic and gravitational time dilation, and length contraction.",
}

MAX_TOKENS = 50
# SERVER_URL = "http://127.0.0.1:8000"
SERVER_URL = "http://localhost:8000"

# ── Single request ───────────────────────────────────────────────
async def send_request(client: httpx.AsyncClient,
                       prompt: str,
                       max_tokens: int,
                       request_id: int,
            ) -> dict:
    
    start = time.perf_counter()
    try:
        response = await client.post(f"{SERVER_URL}/generate",
                                     json = {"prompt": prompt, "max_tokens": max_tokens},
                                     timeout = 120.0)
        
        elapsed = time.perf_counter() - start
        if response.status_code == 200:
            data = response.json()
            return {
                "request_id": request_id,
                "status": "OK",
                "latency_sec": elapsed,
                "token_count": data.get("token_count", 0),
                "prompt_type": None,  # filled by caller
            }
        else:
            return {
                "request_id": request_id,
                "status": f"HTTP {response.status_code}",
                "latency_sec": elapsed,
                "token_count": 0,
                "prompt_type": None,
            }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "request_id": request_id,
            "status": str(e),
            "latency_sec": elapsed,
            "token_count": 0,
            "prompt_type": None,
        }
        
# ── Concurrent requests ───────────────────────────────────────────────
async def run_concurrent_requests(concurrency: int,
                                  prompt_type: str = "short",
            ) -> dict:
    
    prompt = PROMPT[prompt_type]
    async with httpx.AsyncClient(trust_env = False) as client:
        wall_start = time.perf_counter()
        tasks = [
                send_request(client, prompt, MAX_TOKENS, i) 
                for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        wall_elapsed = time.perf_counter() - wall_start
        
    for r in results:
        r["prompt_type"] = prompt_type
        
    ok_results = [r for r in results if r["status"] == "OK"]
    failed = len(results) - len(ok_results)
    
    if not ok_results:
        return {
            "concurrency": concurrency,
            "prompt_type": prompt_type,
            "total_requests": len(results),
            "failed": failed,
            "error": "All requests failed",
        } 
        
    latencies = [r["latency_sec"] for r in ok_results]
    total_tokens = sum(r["token_count"] for r in ok_results)
    
    return {
        "concurrency": concurrency,
        "prompt_type": prompt_type,
        "total_requests": len(results),
        "failed": failed,
        "wall_time_sec": wall_elapsed,
        "p50_sec": float(np.percentile(latencies, 50)),
        "p90_sec": float(np.percentile(latencies, 90)),
        "p95_sec": float(np.percentile(latencies, 95)),
        "p99_sec": float(np.percentile(latencies, 99)),
        "mean_sec": statistics.mean(latencies),
        "std_sec": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_sec": min(latencies),
        "max_sec": max(latencies),
        "total_tokens": total_tokens,
        "agg_tok_per_sec": total_tokens / wall_elapsed if wall_elapsed > 0 else 0,
        "req_per_sec": len(ok_results) / wall_elapsed if wall_elapsed > 0 else 0,
    }
    
def display_results(all_results: List[dict]) -> None:
    
    header = "%-12s %-8s %-9s %-9s %-9s %-9s %-16s %-12s %-8s %-6s" % (
        "Concurrency", "Prompt", "p50 (s)", "p90 (s)", "p95 (s)", "p99 (s)",
        "Mean±Std (s)", "Agg Tok/s", "Req/s", "Fail"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for r in all_results:
        if "error" in r:
            logger.info("%-12s %-8s  ALL FAILED" % (r["concurrency"], r["prompt_type"]))
            continue
        mean_std = "%.3f\u00b1%.3f" % (r["mean_sec"], r["std_sec"])
        row = "%-12s %-8s %-9.3f %-9.3f %-9.3f %-9.3f %-16s %-12.1f %-8.1f %-6d" % (
            r["concurrency"], r["prompt_type"],
            r["p50_sec"], r["p90_sec"], r["p95_sec"], r["p99_sec"],
            mean_std, r["agg_tok_per_sec"], r["req_per_sec"], r["failed"],
        )
        logger.info(row)
        
async def main(concurrency_levels: List[int],
               prompt_type: str,
               out_path: str = None,
        ) -> None:
    
    all_results = []
    for concurrency in concurrency_levels:
        logger.info("Running concurrency=%d, prompt=%s ...", concurrency, prompt_type)
        result = await run_concurrent_requests(concurrency, prompt_type)
        all_results.append(result)
        logger.info("  Done: wall=%.2fs, p50=%.3fs, agg_tok/s=%.1f",
                     result.get("wall_time_sec", 0),
                     result.get("p50_sec", 0),
                     result.get("agg_tok_per_sec", 0))
        
    display_results(all_results)
    
    if out_path:
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Results saved to %s", out_path)
        
def parse_args():
    parser = argparse.ArgumentParser(description="Concurrent load test for LLM serving layer")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[4, 8, 16, 32],
                        help="Concurrency levels to test (default: 4 8 16 32)")
    parser.add_argument("--prompt", type=str, default="short",
                        choices=["short", "medium", "long"],
                        help="Prompt length category (default: short)")
    parser.add_argument("--out", type=str, default=None,
                        help="Save results to JSON file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.concurrency, args.prompt, args.out))