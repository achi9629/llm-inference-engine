import torch, pynvml # type: ignore
from torch.cuda import memory_allocated, memory_reserved, get_device_properties # type: ignore

import logging
logger = logging.getLogger(__name__)

class GPUMonitor:
    def __init__(self, device_id: int = 0) -> None:
        
        '''
        Initializes the GPU monitor for a specific device.
        Args:
            device_id (int): The ID of the GPU device to monitor (default: 0).
        '''
        
        self.device_id = device_id
        self._nvml_available = False
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            self._nvml_available = True
        except Exception as e:
            logger.warning(f"NVML initialization failed: {e}")
            self.handle = None

    def snapshot(self) -> dict:
        
        '''
        Takes a snapshot of the current GPU memory usage and utilization.
        
        Definition:
            - memory_allocated_mb: The amount of GPU memory currently allocated by PyTorch (in MB).
            - memory_reserved_mb: The amount of GPU memory currently reserved by PyTorch (in MB).
            - memory_total_mb: The total amount of GPU memory available on the device (in MB).
            - memory_free_mb: The amount of GPU memory currently free (in MB).
            - gpu_memory_total_mb: The total amount of GPU memory available on the device according to NVML (in MB).
            - gpu_memory_used_mb: The amount of GPU memory currently used according to NVML (in MB).
            - gpu_memory_free_mb: The amount of GPU memory currently free according to NVML (in MB).
            - gpu_utilization_percent: The current GPU utilization percentage according to NVML. 
                                       It does not tell gpu efficiency, but rather how busy the GPU is.
            - memory_utilization_percent: The current memory utilization percentage according to NVML.
        
        Returns:
            dict: A dictionary containing memory usage and utilization metrics.
        '''

        memory_allocated_mb = memory_allocated(self.device_id) / (1024 ** 2)
        memory_reserved_mb = memory_reserved(self.device_id) / (1024 ** 2)
        memory_total_mb = get_device_properties(self.device_id).total_memory / (1024 ** 2)
        memory_free_mb = memory_total_mb - memory_reserved_mb
        
        if self._nvml_available:
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_memory_total_mb = mem_info.total / (1024 ** 2)
            gpu_memory_used_mb = mem_info.used / (1024 ** 2)
            gpu_memory_free_mb = mem_info.free / (1024 ** 2)
            
            return {
                "memory_allocated_mb": memory_allocated_mb,
                "memory_reserved_mb": memory_reserved_mb,
                "memory_total_mb": memory_total_mb,
                "memory_free_mb": memory_free_mb,
                "gpu_memory_total_mb": gpu_memory_total_mb,
                "gpu_memory_used_mb": gpu_memory_used_mb,
                "gpu_memory_free_mb": gpu_memory_free_mb,
                "gpu_utilization_percent": utilization.gpu,
                "memory_utilization_percent": utilization.memory
            }
        else:
            return {
                "memory_allocated_mb": memory_allocated_mb,
                "memory_reserved_mb": memory_reserved_mb,
                "memory_total_mb": memory_total_mb,
                "memory_free_mb": memory_free_mb,
                "gpu_memory_total_mb": None,
                "gpu_memory_used_mb": None,
                "gpu_memory_free_mb": None,
                "gpu_utilization_percent": None,
                "memory_utilization_percent": None
            }
            
    def summary(self) -> str:
        
        '''
        Generates a summary of the current GPU memory usage and utilization.
        
        Returns:
            str: A formatted string summarizing the GPU memory usage and utilization.
        '''
        
        snapshot = self.snapshot()
        
        logger.info(f"GPU Memory Allocated: {snapshot['memory_allocated_mb']:.2f} MB")
        logger.info(f"GPU Memory Reserved: {snapshot['memory_reserved_mb']:.2f} MB")
        logger.info(f"GPU Memory Total: {snapshot['memory_total_mb']:.2f} MB")
        logger.info(f"GPU Memory Free: {snapshot['memory_free_mb']:.2f} MB")
        
        if self._nvml_available:
            logger.info(f"GPU Memory Total (NVML): {snapshot['gpu_memory_total_mb']:.2f} MB")
            logger.info(f"GPU Memory Used (NVML): {snapshot['gpu_memory_used_mb']:.2f} MB")
            logger.info(f"GPU Memory Free (NVML): {snapshot['gpu_memory_free_mb']:.2f} MB")
            logger.info(f"GPU Utilization: {snapshot['gpu_utilization_percent']}%")
            logger.info(f"Memory Utilization: {snapshot['memory_utilization_percent']}%")
            
    def reset_peak_memory_stats(self) -> None:
        
        '''
        Resets the peak memory statistics for the monitored GPU device.
        '''
        
        torch.cuda.reset_peak_memory_stats(self.device_id)