import torch, time
import logging
logger = logging.getLogger(__name__)

from llm_engine import GPUMonitor

class BenchmarkMetrics:
    def __init__(self, device_id: int = 0, num_params: int = 0, dtype: torch.Tensor = torch.float32) -> None:
        
        self.device_id = device_id
        self.num_params = num_params
        self.dtype = dtype
        self.gpu_monitor = GPUMonitor(device_id = self.device_id)
        self._start_time = None
        self._end_time = None
        self._num_tokens = 0
        
    def get_peak_tflops(self) -> float:
        """Returns peak TFLOPS for the detected GPU and dtype."""
    
        # Dense tensor core TFLOPS (source: NVIDIA official specs)
        GPU_PEAK_TFLOPS = {
                            "A100": {"fp32": 19.5, "tf32": 156.0, "fp16": 312.0, "bf16": 312.0},
                            "A10G": {"fp32": 31.2, "tf32": 62.5,  "fp16": 125.0, "bf16": 125.0},
                            "H100": {"fp32": 67.0, "tf32": 990.0, "fp16": 1979.0, "bf16": 1979.0},
                            "H200": {"fp32": 67.0, "tf32": 990.0, "fp16": 1979.0, "bf16": 1979.0},
                            "L40S": {"fp32": 91.6, "tf32": 183.0, "fp16": 362.0, "bf16": 362.0},
                            "V100": {"fp32": 15.7, "tf32": 0.0,   "fp16": 125.0, "bf16": 0.0},
                            "RTX 3090": {"fp32": 35.6, "tf32": 71.0,  "fp16": 142.0, "bf16": 142.0},
                            "RTX 4090": {"fp32": 82.6, "tf32": 165.0, "fp16": 330.0, "bf16": 330.0},
            }
        
        dtype_map = {
                    torch.float32: "fp32",
                    torch.float16: "fp16",
                    torch.bfloat16: "bf16",
            }
        dtype_str = dtype_map[self.dtype]
        
        gpu_name = torch.cuda.get_device_name(self.device_id)
        
        for key, tflops in GPU_PEAK_TFLOPS.items():
            if key.lower() in gpu_name.lower():
                if dtype_str not in tflops:
                    raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose from: fp32, fp16, bf16")
                if tflops[dtype_str] == 0.0:
                    raise ValueError(f"{gpu_name} does not support {dtype_str}")
                return tflops[dtype_str]
            
        raise ValueError(f"Unknown GPU: '{gpu_name}'. Add its specs to GPU_PEAK_TFLOPS.")
        
    def start(self) -> None:
        
        self.gpu_monitor.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self._start_time = time.perf_counter()
        
    def stop(self, num_tokens: int) -> None:
        
        torch.cuda.synchronize()
        self._end_time = time.perf_counter()
        self._num_tokens = num_tokens
        self.gpu_snapshot = self.gpu_monitor.snapshot()
        self.peak_memory_mb = torch.cuda.max_memory_allocated(self.device_id) / (1024 ** 2)
        
    def result(self) -> dict:
        
        latency = self._end_time - self._start_time
        tokens_per_sec = self._num_tokens / latency
        peak_tflops = self.get_peak_tflops()
        mfu = ( 2 * self.num_params * tokens_per_sec ) / ( peak_tflops * 1e12 ) * 100
        
        return {
            "latency_sec": latency,
            "tokens_per_sec": tokens_per_sec,
            "peak_memory_mb": self.peak_memory_mb,
            "gpu_snapshot": self.gpu_snapshot,
            "MFU_percent": mfu
        }
        
    def summary(self):
        
        result = self.result()
        logger.info(f"Latency: {result['latency_sec']:.2f} sec")
        logger.info(f"Tokens per second: {result['tokens_per_sec']:.2f}")
        logger.info(f"Peak GPU Memory: {result['peak_memory_mb']:.2f} MB")
        logger.info(f"GPU Utilization: {result['gpu_snapshot']['gpu_utilization_percent']}%")
        logger.info(f"Memory Utilization: {result['gpu_snapshot']['memory_utilization_percent']}%")
        logger.info(f"MFU: {result['MFU_percent']:.2f}%")
        