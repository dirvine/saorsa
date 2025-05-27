#!/usr/bin/env python3
"""
Metal Performance Shaders (MPS) Optimizer for Saorse Robot System

This module provides Mac M3 specific optimizations for AI models using
Metal Performance Shaders, unified memory, and Neural Engine when available.
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import numpy as np
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    import psutil
    import platform
except ImportError as e:
    logging.error(f"Required MPS optimization libraries not found: {e}")
    raise

console = Console()
logger = logging.getLogger(__name__)

# Suppress MPS warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")


@dataclass
class MPSCapabilities:
    """Mac M3/MPS system capabilities."""
    is_available: bool = False
    chip_type: str = "unknown"
    unified_memory_gb: float = 0.0
    gpu_core_count: int = 0
    neural_engine_available: bool = False
    max_buffer_size: int = 0
    supports_fp16: bool = False
    supports_int8: bool = False
    metal_version: str = "unknown"


@dataclass
class OptimizationProfile:
    """Optimization profile for different use cases."""
    name: str
    description: str
    memory_optimization: bool = True
    speed_optimization: bool = True
    precision: str = "fp16"  # fp32, fp16, int8
    batch_size: int = 1
    max_sequence_length: int = 512
    use_unified_memory: bool = True
    enable_compile: bool = True
    enable_channels_last: bool = True


class MPSProfiler:
    """Profiles MPS performance for model optimization."""
    
    def __init__(self):
        self.capabilities = self._detect_capabilities()
        self.benchmark_history: List[Dict[str, Any]] = []
        
    def _detect_capabilities(self) -> MPSCapabilities:
        """Detect Mac M3/MPS capabilities."""
        
        caps = MPSCapabilities()
        
        # Check MPS availability
        caps.is_available = torch.backends.mps.is_available()
        
        if not caps.is_available:
            return caps
            
        # Detect chip type
        try:
            chip_info = platform.processor()
            if "Apple" in chip_info:
                if "M3" in chip_info:
                    caps.chip_type = "M3"
                    caps.gpu_core_count = self._detect_m3_gpu_cores()
                elif "M2" in chip_info:
                    caps.chip_type = "M2"
                    caps.gpu_core_count = self._detect_m2_gpu_cores()
                elif "M1" in chip_info:
                    caps.chip_type = "M1"
                    caps.gpu_core_count = self._detect_m1_gpu_cores()
        except Exception:
            pass
            
        # Get unified memory info
        memory_info = psutil.virtual_memory()
        caps.unified_memory_gb = memory_info.total / (1024**3)
        
        # Test MPS capabilities
        caps.supports_fp16 = self._test_fp16_support()
        caps.supports_int8 = self._test_int8_support()
        caps.max_buffer_size = self._test_max_buffer_size()
        
        # Neural Engine detection (indirect)
        caps.neural_engine_available = caps.chip_type in ["M1", "M2", "M3"]
        
        return caps
        
    def _detect_m3_gpu_cores(self) -> int:
        """Detect M3 GPU core count."""
        # M3 variants: 8-core (base), 10-core (Pro), 40-core (Max)
        # This is a heuristic based on system specs
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if memory_gb >= 64:  # M3 Max
            return 40
        elif memory_gb >= 32:  # M3 Pro
            return 18
        else:  # M3 Base
            return 10
            
    def _detect_m2_gpu_cores(self) -> int:
        """Detect M2 GPU core count."""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if memory_gb >= 64:  # M2 Max/Ultra
            return 38
        elif memory_gb >= 32:  # M2 Pro
            return 19
        else:  # M2 Base
            return 10
            
    def _detect_m1_gpu_cores(self) -> int:
        """Detect M1 GPU core count."""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if memory_gb >= 64:  # M1 Max/Ultra
            return 32
        elif memory_gb >= 32:  # M1 Pro
            return 16
        else:  # M1 Base
            return 8
            
    def _test_fp16_support(self) -> bool:
        """Test FP16 support on MPS."""
        try:
            device = torch.device("mps")
            x = torch.randn(100, 100, dtype=torch.float16, device=device)
            y = torch.randn(100, 100, dtype=torch.float16, device=device)
            z = torch.mm(x, y)
            return True
        except Exception:
            return False
            
    def _test_int8_support(self) -> bool:
        """Test INT8 support on MPS."""
        try:
            device = torch.device("mps")
            x = torch.randint(0, 127, (100, 100), dtype=torch.int8, device=device)
            y = x + 1
            return True
        except Exception:
            return False
            
    def _test_max_buffer_size(self) -> int:
        """Test maximum buffer size for MPS."""
        try:
            device = torch.device("mps")
            max_size = 0
            
            # Binary search for max size
            low, high = 1024, 8 * 1024 * 1024 * 1024  # 1KB to 8GB
            
            while low <= high:
                mid = (low + high) // 2
                try:
                    x = torch.zeros(mid // 4, dtype=torch.float32, device=device)
                    del x
                    torch.mps.empty_cache()
                    max_size = mid
                    low = mid + 1
                except Exception:
                    high = mid - 1
                    
            return max_size
            
        except Exception:
            return 0
            
    def benchmark_operation(self, operation: str, model: Any = None, 
                          input_shape: Tuple[int, ...] = (1, 512)) -> Dict[str, Any]:
        """Benchmark a specific operation on MPS."""
        
        if not self.capabilities.is_available:
            return {"error": "MPS not available"}
            
        device = torch.device("mps")
        results = {
            "operation": operation,
            "input_shape": input_shape,
            "timestamp": time.time(),
            "device": "mps"
        }
        
        try:
            if operation == "matrix_multiply":
                results.update(self._benchmark_matrix_multiply(device, input_shape))
            elif operation == "convolution":
                results.update(self._benchmark_convolution(device, input_shape))
            elif operation == "attention":
                results.update(self._benchmark_attention(device, input_shape))
            elif operation == "model_inference" and model:
                results.update(self._benchmark_model_inference(model, device, input_shape))
                
        except Exception as e:
            results["error"] = str(e)
            
        self.benchmark_history.append(results)
        return results
        
    def _benchmark_matrix_multiply(self, device: torch.device, 
                                 shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Benchmark matrix multiplication."""
        
        size = shape[0] if len(shape) > 0 else 1024
        
        # Test different precisions
        results = {}
        
        for dtype, dtype_name in [(torch.float32, "fp32"), (torch.float16, "fp16")]:
            try:
                x = torch.randn(size, size, dtype=dtype, device=device)
                y = torch.randn(size, size, dtype=dtype, device=device)
                
                # Warmup
                for _ in range(5):
                    _ = torch.mm(x, y)
                torch.mps.synchronize()
                
                # Benchmark
                start_time = time.time()
                for _ in range(20):
                    z = torch.mm(x, y)
                torch.mps.synchronize()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 20
                gflops = (2 * size**3) / (avg_time * 1e9)
                
                results[f"{dtype_name}_time"] = avg_time
                results[f"{dtype_name}_gflops"] = gflops
                
            except Exception as e:
                results[f"{dtype_name}_error"] = str(e)
                
        return results
        
    def _benchmark_convolution(self, device: torch.device,
                             shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Benchmark convolution operations."""
        
        batch_size = shape[0] if len(shape) > 0 else 1
        
        results = {}
        
        try:
            # 2D Convolution benchmark
            input_tensor = torch.randn(batch_size, 64, 224, 224, device=device)
            conv_layer = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
            
            # Warmup
            for _ in range(5):
                _ = conv_layer(input_tensor)
            torch.mps.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                output = conv_layer(input_tensor)
            torch.mps.synchronize()
            end_time = time.time()
            
            results["conv2d_time"] = (end_time - start_time) / 10
            results["conv2d_throughput"] = batch_size / results["conv2d_time"]
            
        except Exception as e:
            results["conv2d_error"] = str(e)
            
        return results
        
    def _benchmark_attention(self, device: torch.device,
                           shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Benchmark attention operations."""
        
        batch_size = shape[0] if len(shape) > 0 else 1
        seq_len = shape[1] if len(shape) > 1 else 512
        d_model = 768
        
        results = {}
        
        try:
            # Multi-head attention benchmark
            query = torch.randn(batch_size, seq_len, d_model, device=device)
            key = torch.randn(batch_size, seq_len, d_model, device=device)
            value = torch.randn(batch_size, seq_len, d_model, device=device)
            
            attention = nn.MultiheadAttention(d_model, num_heads=12).to(device)
            
            # Warmup
            for _ in range(3):
                _ = attention(query, key, value)
            torch.mps.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(5):
                output, _ = attention(query, key, value)
            torch.mps.synchronize()
            end_time = time.time()
            
            results["attention_time"] = (end_time - start_time) / 5
            results["attention_tokens_per_sec"] = (batch_size * seq_len) / results["attention_time"]
            
        except Exception as e:
            results["attention_error"] = str(e)
            
        return results
        
    def _benchmark_model_inference(self, model: Any, device: torch.device,
                                 input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Benchmark full model inference."""
        
        results = {}
        
        try:
            model = model.to(device)
            model.eval()
            
            # Create dummy input
            if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                # Language model
                input_ids = torch.randint(0, model.config.vocab_size, input_shape, device=device)
                inputs = {"input_ids": input_ids}
            else:
                # Generic model
                inputs = torch.randn(input_shape, device=device)
                
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    if isinstance(inputs, dict):
                        _ = model(**inputs)
                    else:
                        _ = model(inputs)
            torch.mps.synchronize()
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(5):
                    if isinstance(inputs, dict):
                        output = model(**inputs)
                    else:
                        output = model(inputs)
            torch.mps.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 5
            
            results["inference_time"] = avg_time
            
            # Calculate throughput
            if isinstance(inputs, dict):
                batch_size = inputs["input_ids"].shape[0]
                seq_len = inputs["input_ids"].shape[1] if len(inputs["input_ids"].shape) > 1 else 1
            else:
                batch_size = inputs.shape[0]
                seq_len = inputs.shape[1] if len(inputs.shape) > 1 else 1
                
            results["tokens_per_second"] = (batch_size * seq_len) / avg_time
            results["samples_per_second"] = batch_size / avg_time
            
        except Exception as e:
            results["inference_error"] = str(e)
            
        return results


class MPSOptimizer:
    """
    Optimizes models for Metal Performance Shaders on Mac M3.
    
    Provides automatic optimization, manual tuning, and performance monitoring
    for Hugging Face models running on Apple Silicon.
    """
    
    def __init__(self):
        self.profiler = MPSProfiler()
        self.optimization_profiles = self._create_optimization_profiles()
        self.optimized_models: Dict[str, Any] = {}
        
    def _create_optimization_profiles(self) -> Dict[str, OptimizationProfile]:
        """Create optimization profiles for different use cases."""
        
        profiles = {}
        
        # Speed-focused profile
        profiles["speed"] = OptimizationProfile(
            name="Speed",
            description="Maximum inference speed",
            memory_optimization=False,
            speed_optimization=True,
            precision="fp16",
            batch_size=1,
            max_sequence_length=256,
            use_unified_memory=True,
            enable_compile=True,
            enable_channels_last=True
        )
        
        # Memory-focused profile
        profiles["memory"] = OptimizationProfile(
            name="Memory Efficient",
            description="Minimize memory usage",
            memory_optimization=True,
            speed_optimization=False,
            precision="fp16",
            batch_size=1,
            max_sequence_length=512,
            use_unified_memory=True,
            enable_compile=False,
            enable_channels_last=False
        )
        
        # Balanced profile
        profiles["balanced"] = OptimizationProfile(
            name="Balanced",
            description="Balance between speed and memory",
            memory_optimization=True,
            speed_optimization=True,
            precision="fp16",
            batch_size=1,
            max_sequence_length=384,
            use_unified_memory=True,
            enable_compile=True,
            enable_channels_last=True
        )
        
        # Quality-focused profile
        profiles["quality"] = OptimizationProfile(
            name="Quality",
            description="Best output quality",
            memory_optimization=False,
            speed_optimization=False,
            precision="fp32",
            batch_size=1,
            max_sequence_length=1024,
            use_unified_memory=True,
            enable_compile=False,
            enable_channels_last=False
        )
        
        return profiles
        
    def optimize_model(self, model: Any, profile_name: str = "balanced",
                      custom_profile: Optional[OptimizationProfile] = None) -> Any:
        """Optimize a model for MPS using specified profile."""
        
        if not self.profiler.capabilities.is_available:
            console.print("[yellow]MPS not available, returning unoptimized model[/yellow]")
            return model
            
        profile = custom_profile or self.optimization_profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}")
            
        console.print(f"[blue]Optimizing model with '{profile.name}' profile...[/blue]")
        
        try:
            # Move model to MPS device
            device = torch.device("mps")
            optimized_model = model.to(device)
            
            # Apply precision optimization
            if profile.precision == "fp16" and self.profiler.capabilities.supports_fp16:
                optimized_model = optimized_model.half()
            elif profile.precision == "int8" and self.profiler.capabilities.supports_int8:
                # Note: Full int8 quantization requires additional setup
                console.print("[yellow]INT8 optimization requires additional quantization setup[/yellow]")
                
            # Apply memory optimizations
            if profile.memory_optimization:
                optimized_model = self._apply_memory_optimizations(optimized_model)
                
            # Apply speed optimizations
            if profile.speed_optimization:
                optimized_model = self._apply_speed_optimizations(optimized_model, profile)
                
            # Set evaluation mode and optimize for inference
            optimized_model.eval()
            
            # Store optimized model
            model_id = id(model)
            self.optimized_models[model_id] = {
                "model": optimized_model,
                "profile": profile,
                "optimization_time": time.time()
            }
            
            console.print(f"[green]✓ Model optimized for {self.profiler.capabilities.chip_type} MPS[/green]")
            return optimized_model
            
        except Exception as e:
            console.print(f"[red]✗ Optimization failed: {e}[/red]")
            logger.error(f"MPS optimization error: {e}")
            return model
            
    def _apply_memory_optimizations(self, model: Any) -> Any:
        """Apply memory-specific optimizations."""
        
        try:
            # Enable gradient checkpointing for training
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                
            # Optimize attention for memory efficiency
            if hasattr(model, 'config'):
                # This would be model-specific optimization
                pass
                
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            
        return model
        
    def _apply_speed_optimizations(self, model: Any, profile: OptimizationProfile) -> Any:
        """Apply speed-specific optimizations."""
        
        try:
            # Enable channels last memory format for better performance
            if profile.enable_channels_last:
                try:
                    model = model.to(memory_format=torch.channels_last)
                except Exception:
                    pass  # Not all models support channels last
                    
            # Compile model for better performance (PyTorch 2.0+)
            if profile.enable_compile and hasattr(torch, 'compile'):
                try:
                    # Use conservative compilation for stability
                    model = torch.compile(model, mode="default", dynamic=False)
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
                    
            # Optimize for inference
            torch.backends.mps.allow_tf32 = True  # Enable TF32 for faster computation
            
        except Exception as e:
            logger.warning(f"Speed optimization failed: {e}")
            
        return model
        
    def benchmark_model(self, model: Any, input_shape: Tuple[int, ...] = (1, 512)) -> Dict[str, Any]:
        """Comprehensive model benchmarking on MPS."""
        
        if not self.profiler.capabilities.is_available:
            return {"error": "MPS not available"}
            
        console.print("[blue]Running comprehensive MPS benchmark...[/blue]")
        
        results = {
            "system_info": {
                "chip_type": self.profiler.capabilities.chip_type,
                "gpu_cores": self.profiler.capabilities.gpu_core_count,
                "unified_memory_gb": self.profiler.capabilities.unified_memory_gb,
                "mps_capabilities": {
                    "fp16_support": self.profiler.capabilities.supports_fp16,
                    "int8_support": self.profiler.capabilities.supports_int8,
                    "max_buffer_size": self.profiler.capabilities.max_buffer_size
                }
            },
            "benchmarks": {}
        }
        
        # Benchmark unoptimized model
        console.print("  Testing unoptimized model...")
        unopt_results = self.profiler.benchmark_operation(
            "model_inference", model, input_shape
        )
        results["benchmarks"]["unoptimized"] = unopt_results
        
        # Benchmark with different optimization profiles
        for profile_name, profile in self.optimization_profiles.items():
            console.print(f"  Testing {profile.name} profile...")
            
            try:
                optimized_model = self.optimize_model(model, profile_name)
                opt_results = self.profiler.benchmark_operation(
                    "model_inference", optimized_model, input_shape
                )
                results["benchmarks"][profile_name] = opt_results
                
                # Calculate improvement
                if "inference_time" in unopt_results and "inference_time" in opt_results:
                    speedup = unopt_results["inference_time"] / opt_results["inference_time"]
                    opt_results["speedup"] = speedup
                    
            except Exception as e:
                results["benchmarks"][profile_name] = {"error": str(e)}
                
        return results
        
    def get_optimization_recommendations(self, use_case: str = "robotics") -> List[str]:
        """Get optimization recommendations for specific use case."""
        
        capabilities = self.profiler.capabilities
        recommendations = []
        
        if not capabilities.is_available:
            recommendations.append("MPS not available - use CPU optimizations")
            return recommendations
            
        # Chip-specific recommendations
        if capabilities.chip_type == "M3":
            recommendations.append("Use 'speed' profile for M3 - excellent performance")
            if capabilities.unified_memory_gb >= 32:
                recommendations.append("Large unified memory - can use larger models")
        elif capabilities.chip_type == "M2":
            recommendations.append("Use 'balanced' profile for M2")
        elif capabilities.chip_type == "M1":
            recommendations.append("Use 'memory' profile for M1 - prioritize efficiency")
            
        # Use case specific
        if use_case == "robotics":
            recommendations.extend([
                "Use small batch sizes (1) for real-time response",
                "Enable FP16 for faster inference",
                "Consider sequence length limits for memory efficiency"
            ])
        elif use_case == "development":
            recommendations.extend([
                "Use 'quality' profile for testing",
                "Enable model compilation for repeated inference"
            ])
            
        # Memory-based recommendations
        if capabilities.unified_memory_gb < 16:
            recommendations.append("Limited memory - use smaller models or 'memory' profile")
        elif capabilities.unified_memory_gb >= 64:
            recommendations.append("High memory - can run large models with 'quality' profile")
            
        return recommendations
        
    def create_custom_profile(self, name: str, **kwargs) -> OptimizationProfile:
        """Create a custom optimization profile."""
        
        profile = OptimizationProfile(name=name, description=kwargs.get("description", "Custom profile"))
        
        # Update profile with provided parameters
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
                
        return profile
        
    def show_capabilities(self) -> Table:
        """Show detailed MPS capabilities."""
        
        table = Table(title="Mac M3 MPS Capabilities")
        table.add_column("Feature", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        caps = self.profiler.capabilities
        
        table.add_row("MPS Available", str(caps.is_available), 
                     "✓" if caps.is_available else "✗")
        table.add_row("Chip Type", caps.chip_type, "")
        table.add_row("GPU Cores", str(caps.gpu_core_count), "")
        table.add_row("Unified Memory", f"{caps.unified_memory_gb:.1f}GB", "")
        table.add_row("FP16 Support", str(caps.supports_fp16),
                     "✓" if caps.supports_fp16 else "✗")
        table.add_row("INT8 Support", str(caps.supports_int8),
                     "✓" if caps.supports_int8 else "✗")
        table.add_row("Max Buffer Size", f"{caps.max_buffer_size / (1024**3):.1f}GB", "")
        table.add_row("Neural Engine", str(caps.neural_engine_available),
                     "✓" if caps.neural_engine_available else "✗")
                     
        return table
        
    def export_benchmark_results(self, filepath: str):
        """Export benchmark results to JSON file."""
        
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "capabilities": {
                        "is_available": self.profiler.capabilities.is_available,
                        "chip_type": self.profiler.capabilities.chip_type,
                        "gpu_core_count": self.profiler.capabilities.gpu_core_count,
                        "unified_memory_gb": self.profiler.capabilities.unified_memory_gb,
                        "supports_fp16": self.profiler.capabilities.supports_fp16,
                        "supports_int8": self.profiler.capabilities.supports_int8
                    },
                    "benchmark_history": self.profiler.benchmark_history
                }, f, indent=2)
                
            console.print(f"[green]✓ Benchmark results exported to {filepath}[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Failed to export results: {e}[/red]")


async def main():
    """Example usage of MPS Optimizer."""
    
    optimizer = MPSOptimizer()
    
    console.print("[blue]Saorse MPS Optimizer Demo[/blue]\n")
    
    # Show capabilities
    capabilities_table = optimizer.show_capabilities()
    console.print(capabilities_table)
    
    # Show recommendations
    recommendations = optimizer.get_optimization_recommendations("robotics")
    console.print("\n[blue]Optimization Recommendations:[/blue]")
    for rec in recommendations:
        console.print(f"  • {rec}")
        
    # Test basic MPS operations
    if optimizer.profiler.capabilities.is_available:
        console.print("\n[blue]Running MPS benchmarks...[/blue]")
        
        # Matrix multiplication benchmark
        mm_results = optimizer.profiler.benchmark_operation("matrix_multiply", input_shape=(1024,))
        if "fp16_gflops" in mm_results:
            console.print(f"Matrix Multiply (FP16): {mm_results['fp16_gflops']:.1f} GFLOPS")
            
        # Attention benchmark
        attn_results = optimizer.profiler.benchmark_operation("attention", input_shape=(1, 512))
        if "attention_tokens_per_sec" in attn_results:
            console.print(f"Attention: {attn_results['attention_tokens_per_sec']:.0f} tokens/sec")
            
    else:
        console.print("\n[yellow]MPS not available - skipping benchmarks[/yellow]")
        
    # Show available optimization profiles
    console.print("\n[blue]Available Optimization Profiles:[/blue]")
    for name, profile in optimizer.optimization_profiles.items():
        console.print(f"  • {profile.name}: {profile.description}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())