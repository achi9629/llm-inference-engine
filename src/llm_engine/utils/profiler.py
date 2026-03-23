import torch
import torch.profiler

class InferenceProfiler:
    def __init__(self,
                 use_cuda: bool = False,
                 record_shapes: bool = False,
                 with_stack: bool = False,
        ) -> None:
        
        '''
        Description:
            Initialize the InferenceProfiler with the specified settings.
        Args:
            use_cuda (bool): Whether to include CUDA activities in the profiling. Default is False.
            record_shapes (bool): Whether to record the shapes of the tensors involved in the operations. Default is False.
            with_stack (bool): Whether to include stack traces in the profiling results. Default is False.
        '''
        
        self.use_cuda = use_cuda
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        
    def profile(self) -> torch.profiler.profile:
        
        '''
        Description:
            Create a PyTorch profiler context manager that can be used to profile the inference process.
        Returns:
            torch.profiler.profile: A context manager for profiling PyTorch operations.
        '''
        
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.use_cuda and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        self._prof = torch.profiler.profile(
                                      activities = activities,
                                      record_shapes = self.record_shapes,
                                      with_stack = self.with_stack,
                                      profile_memory = True
            )
        return self._prof
        
    def summary(self, 
                top_n: int = 20, 
                sort_by: str = 'cpu_time_total'
        ) -> str:
        
        '''
        Description:
            Generate a summary of the profiling results, sorted by the specified metric.
            
        Args:
            top_n (int): The number of top results to display.
            sort_by (str): The metric to sort by. 
            Options include 'cpu_time_total' -> Total CPU time spent on the operation,
                            'cuda_time_total' -> Total CUDA time spent on the operation,
                            'self_cpu_time_total' -> Total CPU time spent on the operation itself (excluding child operations),
                            'self_cuda_time_total' -> Total CUDA time spent on the operation itself (excluding child operations),
                            'cpu_memory_usage' -> Total CPU memory usage for the operation,
                            'cuda_memory_usage' -> Total CUDA memory usage for the operation.
        Returns:
            str: A formatted string containing the profiling summary.
        '''
        
        return self._prof.key_averages().table(sort_by = sort_by, row_limit = top_n)
    
    def export_trace(self, path: str) -> None:
        
        '''
        Description:
            Export the profiling results to a file in a format that can be visualized using tools like TensorBoard.
            
        Args:
            path (str): The file path where the profiling results should be saved.
        '''
        
        self._prof.export_chrome_trace(path)