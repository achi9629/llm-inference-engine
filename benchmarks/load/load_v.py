"""
Day 19 remaining benchmarks:
  1. Queue depth / backpressure test (100 requests, max_batch=4)
  2. Request arrival patterns (burst vs steady vs Poisson)
  3. GPU utilization and memory recording during load
"""

import asyncio, time, logging, argparse, json, threading
import numpy as np
import httpx, pynvml
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

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
SERVER_URL = "http://localhost:8000"


# ── GPU Monitor (background thread) ─────────────────────────────────
class GPUPoller:
    """Polls GPU metrics via pynvml in a background thread."""

    def __init__(self, 
                 device_id: int = 0, 
                 interval: float = 0.25
        ) -> None:
        self.device_id = device_id
        self.interval = interval
        self.samples: List[dict] = []
        self._stop = threading.Event()

    def _poll(self) -> None:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        while not self._stop.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.samples.append({
                                "timestamp": time.perf_counter(),
                                "gpu_util_pct": util.gpu,
                                "mem_util_pct": util.memory,
                                "mem_used_mb": mem.used / (1024 ** 2),
                                "mem_total_mb": mem.total / (1024 ** 2),
                            })
            self._stop.wait(self.interval)
        pynvml.nvmlShutdown()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=5)
        if not self.samples:
            return {}
        gpu_utils = [s["gpu_util_pct"] for s in self.samples]
        mem_useds = [s["mem_used_mb"] for s in self.samples]
        return {
            "num_samples": len(self.samples),
            "gpu_util_mean": float(np.mean(gpu_utils)),
            "gpu_util_max": float(np.max(gpu_utils)),
            "gpu_util_min": float(np.min(gpu_utils)),
            "mem_used_mean_mb": float(np.mean(mem_useds)),
            "mem_used_max_mb": float(np.max(mem_useds)),
            "mem_used_min_mb": float(np.min(mem_useds)),
            "mem_total_mb": self.samples[0]["mem_total_mb"],
        }


# ── Single request ───────────────────────────────────────────────────
async def send_request(client: httpx.AsyncClient,
                       prompt: str,
                       request_id: int,
    ) -> dict:
    start = time.perf_counter()
    try:
        response = await client.post(f"{SERVER_URL}/generate",
                                     json={"prompt": prompt, "max_tokens": MAX_TOKENS},
                                     timeout=300.0)
        elapsed = time.perf_counter() - start
        if response.status_code == 200:
            data = response.json()
            return {
                    "request_id": request_id, 
                    "status": "OK",
                    "latency_sec": elapsed,
                    "token_count": data.get("token_count", 0)
                }
        return {
                "request_id": request_id, 
                "status": f"HTTP {response.status_code}",
                "latency_sec": elapsed, 
                "token_count": 0
            }
    except Exception as e:
        return {
                "request_id": request_id, 
                "status": str(e),
                "latency_sec": time.perf_counter() - start, 
                "token_count": 0
            }


# ── Arrival pattern helpers ──────────────────────────────────────────
async def arrival_burst(client, prompt, n):
    """All n requests fired simultaneously."""
    tasks = [send_request(client, prompt, i) for i in range(n)]
    return await asyncio.gather(*tasks)


async def arrival_steady(client, prompt, n, interval: float):
    """Requests spaced uniformly by `interval` seconds."""
    tasks = []
    for i in range(n):
        tasks.append(asyncio.ensure_future(send_request(client, prompt, i)))
        if i < n - 1:
            await asyncio.sleep(interval)
    return await asyncio.gather(*tasks)


async def arrival_poisson(client, prompt, n, rate: float):
    """Requests with Poisson (exponential inter-arrival) at `rate` req/s."""
    rng = np.random.default_rng(42)
    tasks = []
    for i in range(n):
        tasks.append(asyncio.ensure_future(send_request(client, prompt, i)))
        if i < n - 1:
            delay = float(rng.exponential(1.0 / rate))
            await asyncio.sleep(delay)
    return await asyncio.gather(*tasks)


def summarize(results: list, wall_time: float, pattern: str, gpu_stats: dict) -> dict:
    ok = [r for r in results if r["status"] == "OK"]
    failed = len(results) - len(ok)
    if not ok:
        return {
                "pattern": pattern, 
                "total": len(results), 
                "failed": failed,
                "error": "all failed", 
                "gpu": gpu_stats
            }
        
    latencies = [r["latency_sec"] for r in ok]
    
    total_tokens = sum(r["token_count"] for r in ok)
    
    return {
            "pattern": pattern,
            "total_requests": len(results),
            "failed": failed,
            "wall_time_sec": wall_time,
            "p50_sec": float(np.percentile(latencies, 50)),
            "p90_sec": float(np.percentile(latencies, 90)),
            "p95_sec": float(np.percentile(latencies, 95)),
            "p99_sec": float(np.percentile(latencies, 99)),
            "mean_sec": float(np.mean(latencies)),
            "std_sec": float(np.std(latencies)),
            "min_sec": float(np.min(latencies)),
            "max_sec": float(np.max(latencies)),
            "total_tokens": total_tokens,
            "agg_tok_per_sec": total_tokens / wall_time,
            "req_per_sec": len(ok) / wall_time,
            "gpu": gpu_stats,
    }


# ── Main benchmark runner ────────────────────────────────────────────
async def run_benchmark(num_requests: int,
                        prompt_type: str,
                        device_id: int,
                        steady_interval: float,
                        poisson_rate: float,
                        out_path: str | None,
    ):
    prompt = PROMPT[prompt_type]
    all_results = []

    patterns = [
        ("burst", lambda c, p: arrival_burst(c, p, num_requests)),
        ("steady", lambda c, p: arrival_steady(c, p, num_requests, steady_interval)),
        ("poisson", lambda c, p: arrival_poisson(c, p, num_requests, poisson_rate)),
    ]

    for name, fn in patterns:
        logger.info(f"── {name.upper()} pattern: {num_requests} requests, prompt={prompt_type} ──")
        poller = GPUPoller(device_id=device_id, interval=0.25)
        poller.start()

        async with httpx.AsyncClient(trust_env=False) as client:
            wall_start = time.perf_counter()
            results = await fn(client, prompt)
            wall_time = time.perf_counter() - wall_start

        gpu_stats = poller.stop()
        summary = summarize(list(results), wall_time, name, gpu_stats)
        all_results.append(summary)

        logger.info(f"  wall_time={wall_time:.2f}s  ok={summary['total_requests'] - summary['failed']}"
                     f"  failed={summary['failed']}  tok/s={summary['agg_tok_per_sec']:.1f}"
                     f"  p50={summary['p50_sec']:.3f}s  p99={summary['p99_sec']:.3f}s")
        if gpu_stats:
            logger.info(f"  GPU util: mean={gpu_stats['gpu_util_mean']:.1f}%"
                         f"  max={gpu_stats['gpu_util_max']:.0f}%"
                         f"  mem_used: mean={gpu_stats['mem_used_mean_mb']:.0f}MB"
                         f"  max={gpu_stats['mem_used_max_mb']:.0f}MB"
                         f"  / {gpu_stats['mem_total_mb']:.0f}MB")

    # Display comparison table
    logger.info("")
    logger.info(f"{'Pattern':<10} {'Requests':<10} {'Failed':<8} {'Wall (s)':<10} "
                f"{'tok/s':<10} {'req/s':<8} {'p50 (s)':<10} {'p99 (s)':<10} "
                f"{'GPU util%':<10} {'Mem MB':<10}")
    logger.info("-" * 106)
    for r in all_results:
        gpu = r.get("gpu", {})
        logger.info(f"{r['pattern']:<10} {r['total_requests']:<10} {r['failed']:<8} "
                     f"{r['wall_time_sec']:<10.2f} {r['agg_tok_per_sec']:<10.1f} "
                     f"{r['req_per_sec']:<8.1f} {r['p50_sec']:<10.3f} {r['p99_sec']:<10.3f} "
                     f"{gpu.get('gpu_util_mean', 0):<10.1f} {gpu.get('mem_used_max_mb', 0):<10.0f}")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Backpressure + arrival pattern + GPU monitoring benchmark")
    p.add_argument("--num_requests", type=int, default=100, help="Total requests to send (default: 100)")
    p.add_argument("--prompt", type=str, default="short", choices=["short", "medium", "long"])
    p.add_argument("--device_id", type=int, default=0, help="GPU device index for monitoring")
    p.add_argument("--steady_interval", type=float, default=0.1, help="Seconds between requests for steady pattern")
    p.add_argument("--poisson_rate", type=float, default=10.0, help="Mean requests/sec for Poisson pattern")
    p.add_argument("--out", type=str, default=None, help="Path to save JSON results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_benchmark(
        num_requests = args.num_requests,
        prompt_type = args.prompt,
        device_id = args.device_id,
        steady_interval = args.steady_interval,
        poisson_rate = args.poisson_rate,
        out_path = args.out,
    ))