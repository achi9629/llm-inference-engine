"""
Run all benchmarks end-to-end.

Executes each benchmark script as a subprocess so that GPU memory is fully
released between suites. Skips server-dependent load tests (load_test.py,
load_v.py) which require a running server.

Usage:
    PYTHONPATH=. python scripts/run_benchmark.py              # run all
    PYTHONPATH=. python scripts/run_benchmark.py --suite latency
    PYTHONPATH=. python scripts/run_benchmark.py --suite throughput
    PYTHONPATH=. python scripts/run_benchmark.py --suite profiler
    PYTHONPATH=. python scripts/run_benchmark.py --suite load
    PYTHONPATH=. python scripts/run_benchmark.py --list
"""

import argparse, subprocess, sys, time, logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

SUITES = {
    "latency": [
        [sys.executable, "-W", "ignore::DeprecationWarning", "benchmarks/latency/latency_benchmark.py"],
        [sys.executable, "-W", "ignore::DeprecationWarning", "benchmarks/latency/latency_test.py"],
    ],
    "throughput": [
        [sys.executable, "-W", "ignore::DeprecationWarning", "benchmarks/throughput/throughput_benchmark.py"],
        [sys.executable, "-W", "ignore::DeprecationWarning", "benchmarks/throughput/continuous_batching_benchmark.py"],
        [sys.executable, "-W", "ignore::DeprecationWarning", "benchmarks/throughput/paged_kv_cache_benchmark.py"],
    ],
    "profiler": [
        [sys.executable, "-B", "-W", "ignore::DeprecationWarning", "benchmarks/profiler/profiler_benchmark.py"],
    ],
    "load": [
        [sys.executable, "benchmarks/load/load_benchmark.py", "--mode", "run", "--cache_type", "standard", "--out", "benchmarks/load/standard.json"],
        [sys.executable, "benchmarks/load/load_benchmark.py", "--mode", "run", "--cache_type", "paged", "--out", "benchmarks/load/paged.json"],
        [sys.executable, "benchmarks/load/load_benchmark.py", "--mode", "compare", "--standard_json", "benchmarks/load/standard.json", "--paged_json", "benchmarks/load/paged.json"],
    ],
}

SUITE_ORDER = ["latency", "throughput", "profiler", "load"]

def run_command(cmd: list[str]) -> bool:
    label = " ".join(cmd[1:])
    logger.info(f"{'='*80}")
    logger.info(f"Running: {label}")
    logger.info(f"{'='*80}")
    start = time.time()
    result = subprocess.run(cmd, env={"PYTHONPATH": "."})
    elapsed = time.time() - start
    if result.returncode != 0:
        logger.error(f"FAILED ({result.returncode}) in {elapsed:.1f}s: {label}")
        return False
    logger.info(f"Completed in {elapsed:.1f}s: {label}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all benchmarks end-to-end")
    parser.add_argument("--suite", choices=SUITE_ORDER, default=None,
                        help="Run only this benchmark suite (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List available suites and their commands, then exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        for name in SUITE_ORDER:
            print(f"\n[{name}]")
            for cmd in SUITES[name]:
                print(f"  {' '.join(cmd)}")
        sys.exit(0)

    suites_to_run = [args.suite] if args.suite else SUITE_ORDER
    total_start = time.time()
    passed, failed = 0, 0

    for suite in suites_to_run:
        logger.info(f"\n{'#'*80}")
        logger.info(f"# Suite: {suite}")
        logger.info(f"{'#'*80}")
        for cmd in SUITES[suite]:
            if run_command(cmd):
                passed += 1
            else:
                failed += 1

    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*80}")
    logger.info(f"All done: {passed} passed, {failed} failed in {total_elapsed:.1f}s")
    logger.info(f"{'='*80}")
    sys.exit(1 if failed else 0)