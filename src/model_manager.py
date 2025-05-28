#!/usr/bin/env python3
"""
Model Manager for Saorsa Robot System

This module provides comprehensive model management including downloading,
loading, switching, and optimization of local Hugging Face models for
Mac M3 performance.
"""

import asyncio
import logging
import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoConfig,
        pipeline, Pipeline
    )
    from accelerate import infer_auto_device_map, init_empty_weights
    from huggingface_hub import snapshot_download, list_repo_files
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.panel import Panel
except ImportError as e:
    logging.error(f"Required model management libraries not found: {e}")
    raise

console = Console()
logger = logging.getLogger(__name__)


class ModelSize(Enum):
    """Model size categories."""
    TINY = "tiny"      # < 200M parameters
    SMALL = "small"    # 200M - 1B parameters  
    MEDIUM = "medium"  # 1B - 5B parameters
    LARGE = "large"    # 5B - 15B parameters
    XLARGE = "xlarge"  # > 15B parameters


class ModelStatus(Enum):
    """Model status."""
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    INCOMPATIBLE = "incompatible"


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    model_id: str
    description: str
    size: ModelSize
    parameters: Optional[int] = None
    disk_size_mb: Optional[int] = None
    memory_usage_mb: Optional[int] = None
    supported_tasks: List[str] = field(default_factory=list)
    local_path: Optional[Path] = None
    status: ModelStatus = ModelStatus.NOT_DOWNLOADED
    download_progress: float = 0.0
    last_used: Optional[float] = None
    performance_score: Optional[float] = None
    compatibility_score: Optional[float] = None


class ModelRegistry:
    """Registry of available models for robot control."""
    
    MODELS = {
        # LeRobot Models
        "lerobot/pi0": ModelInfo(
            name="Pi-Zero",
            model_id="lerobot/pi0", 
            description="Physical Intelligence foundation model for robotics",
            size=ModelSize.LARGE,
            parameters=7_000_000_000,
            supported_tasks=["robotics", "vision-language-action"],
        ),
        
        # SmolLM2 Series
        "HuggingFaceTB/SmolLM2-135M-Instruct": ModelInfo(
            name="SmolLM2-135M",
            model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            description="Ultra-compact instruction-following model",
            size=ModelSize.TINY,
            parameters=135_000_000,
            supported_tasks=["text-generation", "instruction-following"],
        ),
        
        "HuggingFaceTB/SmolLM2-360M-Instruct": ModelInfo(
            name="SmolLM2-360M", 
            model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
            description="Compact instruction-following model with good performance",
            size=ModelSize.SMALL,
            parameters=360_000_000,
            supported_tasks=["text-generation", "instruction-following"],
        ),
        
        "HuggingFaceTB/SmolLM2-1.7B-Instruct": ModelInfo(
            name="SmolLM2-1.7B",
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", 
            description="Larger SmolLM2 model with enhanced capabilities",
            size=ModelSize.MEDIUM,
            parameters=1_700_000_000,
            supported_tasks=["text-generation", "instruction-following"],
        ),
        
        # Qwen2.5 Series
        "Qwen/Qwen2.5-1.5B-Instruct": ModelInfo(
            name="Qwen2.5-1.5B",
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            description="Multilingual model with strong reasoning capabilities",
            size=ModelSize.MEDIUM,
            parameters=1_500_000_000,
            supported_tasks=["text-generation", "instruction-following", "reasoning"],
        ),
        
        "Qwen/Qwen2.5-3B-Instruct": ModelInfo(
            name="Qwen2.5-3B", 
            model_id="Qwen/Qwen2.5-3B-Instruct",
            description="Larger Qwen model with enhanced capabilities",
            size=ModelSize.MEDIUM,
            parameters=3_000_000_000,
            supported_tasks=["text-generation", "instruction-following", "reasoning"],
        ),
        
        # Phi-3 Series
        "microsoft/Phi-3-mini-4k-instruct": ModelInfo(
            name="Phi-3-Mini",
            model_id="microsoft/Phi-3-mini-4k-instruct",
            description="Microsoft's compact high-performance model", 
            size=ModelSize.MEDIUM,
            parameters=3_800_000_000,
            supported_tasks=["text-generation", "instruction-following", "reasoning"],
        ),
        
        # Gemma Series
        "google/gemma-2-2b-it": ModelInfo(
            name="Gemma-2-2B",
            model_id="google/gemma-2-2b-it",
            description="Google's efficient instruction-tuned model",
            size=ModelSize.MEDIUM, 
            parameters=2_000_000_000,
            supported_tasks=["text-generation", "instruction-following"],
        ),
    }
    
    @classmethod
    def get_model_info(cls, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID."""
        return cls.MODELS.get(model_id)
        
    @classmethod
    def list_models(cls, size_filter: Optional[ModelSize] = None,
                   task_filter: Optional[str] = None) -> List[ModelInfo]:
        """List models with optional filtering."""
        models = list(cls.MODELS.values())
        
        if size_filter:
            models = [m for m in models if m.size == size_filter]
            
        if task_filter:
            models = [m for m in models if task_filter in m.supported_tasks]
            
        return models
        
    @classmethod
    def recommend_models(cls, available_memory_gb: float,
                        device: str = "mps") -> List[ModelInfo]:
        """Recommend models based on system capabilities."""
        models = list(cls.MODELS.values())
        
        # Filter by memory constraints
        recommended = []
        for model in models:
            estimated_memory = cls._estimate_memory_usage(model, device)
            if estimated_memory <= available_memory_gb * 1024:  # Convert to MB
                recommended.append(model)
                
        # Sort by performance/size ratio
        recommended.sort(key=lambda m: (m.size.value, -m.parameters if m.parameters else 0))
        
        return recommended[:5]  # Return top 5
        
    @classmethod
    def _estimate_memory_usage(cls, model: ModelInfo, device: str) -> float:
        """Estimate memory usage in MB."""
        if not model.parameters:
            return 1000  # Default estimate
            
        # Rough estimate: 2 bytes per parameter for float16, plus overhead
        base_memory = (model.parameters * 2) / (1024 * 1024)  # MB
        
        # Add overhead for different devices
        overhead_factor = {
            "mps": 1.3,   # MPS has some overhead
            "cuda": 1.2,  # CUDA is efficient
            "cpu": 1.5    # CPU needs more overhead
        }.get(device, 1.4)
        
        return base_memory * overhead_factor


class ModelDownloader:
    """Handles model downloading from Hugging Face Hub."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    async def download_model(self, model_info: ModelInfo, 
                           force: bool = False) -> bool:
        """Download a model from Hugging Face Hub."""
        
        model_info.status = ModelStatus.DOWNLOADING
        model_info.download_progress = 0.0
        
        try:
            # Check if already downloaded
            local_path = self.cache_dir / model_info.model_id.replace("/", "_")
            
            if local_path.exists() and not force:
                console.print(f"[yellow]Model {model_info.name} already downloaded[/yellow]")
                model_info.local_path = local_path
                model_info.status = ModelStatus.DOWNLOADED
                return True
                
            console.print(f"[blue]Downloading {model_info.name}...[/blue]")
            
            # Download with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                
                download_task = progress.add_task(
                    f"Downloading {model_info.name}...", 
                    total=100
                )
                
                # Download model files
                downloaded_path = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: snapshot_download(
                        model_info.model_id,
                        cache_dir=str(self.cache_dir),
                        local_dir=str(local_path),
                        local_dir_use_symlinks=False
                    )
                )
                
                progress.update(download_task, completed=100)
                
            model_info.local_path = Path(downloaded_path)
            model_info.status = ModelStatus.DOWNLOADED
            
            # Calculate disk size
            model_info.disk_size_mb = self._calculate_directory_size(local_path)
            
            console.print(f"[green]✓ Downloaded {model_info.name} ({model_info.disk_size_mb}MB)[/green]")
            return True
            
        except Exception as e:
            model_info.status = ModelStatus.ERROR
            console.print(f"[red]✗ Failed to download {model_info.name}: {e}[/red]")
            logger.error(f"Download error: {e}")
            return False
            
    def _calculate_directory_size(self, path: Path) -> int:
        """Calculate total size of directory in MB."""
        total_size = 0
        
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                
        return int(total_size / (1024 * 1024))  # Convert to MB
        
    def delete_model(self, model_info: ModelInfo) -> bool:
        """Delete a downloaded model."""
        
        if model_info.local_path and model_info.local_path.exists():
            try:
                shutil.rmtree(model_info.local_path)
                model_info.local_path = None
                model_info.status = ModelStatus.NOT_DOWNLOADED
                console.print(f"[green]✓ Deleted {model_info.name}[/green]")
                return True
            except Exception as e:
                console.print(f"[red]✗ Failed to delete {model_info.name}: {e}[/red]")
                return False
                
        return False


class ModelLoader:
    """Handles model loading and optimization."""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_tokenizers: Dict[str, Any] = {}
        self.loaded_pipelines: Dict[str, Pipeline] = {}
        
        # Determine best device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                
    async def load_model(self, model_info: ModelInfo, 
                        optimization_level: str = "balanced") -> bool:
        """Load a model with specified optimizations."""
        
        if model_info.status != ModelStatus.DOWNLOADED:
            console.print(f"[red]Model {model_info.name} not downloaded[/red]")
            return False
            
        model_info.status = ModelStatus.LOADING
        
        try:
            console.print(f"[blue]Loading {model_info.name} on {self.device}...[/blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                
                load_task = progress.add_task(f"Loading {model_info.name}...", total=None)
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_info.local_path),
                    trust_remote_code=True,
                    local_files_only=True
                )
                
                # Ensure pad token exists
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                # Load model with optimizations
                model = await self._load_optimized_model(
                    model_info, optimization_level
                )
                
                # Create pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
                )
                
                progress.update(load_task, completed=True)
                
            # Store loaded components
            self.loaded_models[model_info.model_id] = model
            self.loaded_tokenizers[model_info.model_id] = tokenizer
            self.loaded_pipelines[model_info.model_id] = pipe
            
            model_info.status = ModelStatus.LOADED
            model_info.last_used = time.time()
            
            # Estimate memory usage
            model_info.memory_usage_mb = self._estimate_loaded_memory(model)
            
            console.print(f"[green]✓ Loaded {model_info.name} ({model_info.memory_usage_mb}MB)[/green]")
            return True
            
        except Exception as e:
            model_info.status = ModelStatus.ERROR
            console.print(f"[red]✗ Failed to load {model_info.name}: {e}[/red]")
            logger.error(f"Model loading error: {e}")
            return False
            
    async def _load_optimized_model(self, model_info: ModelInfo, 
                                   optimization_level: str) -> Any:
        """Load model with device-specific optimizations."""
        
        # Base loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        
        # Optimization based on device and level
        if self.device == "mps":
            model_kwargs.update({
                "torch_dtype": torch.float16,
                "device_map": None,  # Manual device placement for MPS
                "low_cpu_mem_usage": True
            })
        elif self.device == "cuda":
            model_kwargs.update({
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "load_in_8bit": optimization_level == "memory_efficient",
                "load_in_4bit": optimization_level == "ultra_efficient"
            })
        else:  # CPU
            model_kwargs.update({
                "torch_dtype": torch.float32,
                "device_map": None
            })
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(model_info.local_path),
            **model_kwargs
        )
        
        # Move to device if needed
        if self.device == "mps":
            try:
                model = model.to(self.device)
            except Exception as e:
                logger.warning(f"Could not move to MPS, using CPU: {e}")
                self.device = "cpu" 
                model = model.to("cpu")
                
        # Apply optimizations
        if optimization_level == "performance":
            model = self._apply_performance_optimizations(model)
        elif optimization_level == "memory_efficient":
            model = self._apply_memory_optimizations(model)
            
        return model
        
    def _apply_performance_optimizations(self, model) -> Any:
        """Apply performance optimizations."""
        
        try:
            # Compile model for better performance (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="default")
                
            # Enable evaluation mode optimizations
            model.eval()
            
            # For MPS, ensure proper memory format
            if self.device == "mps":
                model = model.to(memory_format=torch.channels_last)
                
        except Exception as e:
            logger.warning(f"Performance optimization failed: {e}")
            
        return model
        
    def _apply_memory_optimizations(self, model) -> Any:
        """Apply memory optimizations."""
        
        try:
            # Gradient checkpointing
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                
            # Half precision for inference
            if self.device != "cpu":
                model = model.half()
                
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            
        return model
        
    def _estimate_loaded_memory(self, model) -> int:
        """Estimate memory usage of loaded model in MB."""
        
        try:
            total_params = sum(p.numel() for p in model.parameters())
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            
            total_size_mb = (param_size + buffer_size) / (1024 * 1024)
            return int(total_size_mb)
            
        except Exception:
            return 0
            
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        
        try:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
            if model_id in self.loaded_tokenizers:
                del self.loaded_tokenizers[model_id]
            if model_id in self.loaded_pipelines:
                del self.loaded_pipelines[model_id]
                
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
                
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False
            
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model IDs."""
        return list(self.loaded_models.keys())


class ModelManager:
    """
    Comprehensive model manager for Saorsa robot system.
    
    Handles model discovery, downloading, loading, switching, and optimization
    with special focus on Mac M3 performance.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry = ModelRegistry()
        self.downloader = ModelDownloader(self.cache_dir)
        self.loader = ModelLoader()
        
        # Model status tracking
        self.model_status: Dict[str, ModelInfo] = {}
        self.current_model: Optional[str] = None
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load cached model information
        self._load_model_cache()
        
    def list_available_models(self, show_status: bool = True) -> Table:
        """List all available models with status."""
        
        table = Table(title="Available Models for Saorsa")
        table.add_column("Name", style="cyan")
        table.add_column("Model ID", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Parameters", style="magenta")
        table.add_column("Status", style="blue")
        table.add_column("Memory", style="red")
        
        for model_info in ModelRegistry.MODELS.values():
            # Get current status
            current_info = self.model_status.get(model_info.model_id, model_info)
            
            # Format parameters
            param_str = "Unknown"
            if current_info.parameters:
                if current_info.parameters >= 1_000_000_000:
                    param_str = f"{current_info.parameters / 1_000_000_000:.1f}B"
                else:
                    param_str = f"{current_info.parameters / 1_000_000:.0f}M"
                    
            # Format memory usage
            memory_str = "N/A"
            if current_info.memory_usage_mb:
                memory_str = f"{current_info.memory_usage_mb}MB"
            elif current_info.disk_size_mb:
                memory_str = f"~{current_info.disk_size_mb}MB"
                
            # Status with colors
            status_str = current_info.status.value
            if current_info.status == ModelStatus.LOADED:
                status_str = f"[green]{status_str}[/green]"
            elif current_info.status == ModelStatus.DOWNLOADED:
                status_str = f"[yellow]{status_str}[/yellow]"
            elif current_info.status == ModelStatus.ERROR:
                status_str = f"[red]{status_str}[/red]"
                
            table.add_row(
                current_info.name,
                current_info.model_id,
                current_info.size.value,
                param_str,
                status_str,
                memory_str
            )
            
        return table
        
    async def download_model(self, model_id: str, force: bool = False) -> bool:
        """Download a specific model."""
        
        model_info = ModelRegistry.get_model_info(model_id)
        if not model_info:
            console.print(f"[red]Unknown model: {model_id}[/red]")
            return False
            
        # Update our tracking
        if model_id not in self.model_status:
            self.model_status[model_id] = model_info
            
        success = await self.downloader.download_model(
            self.model_status[model_id], force
        )
        
        if success:
            self._save_model_cache()
            
        return success
        
    async def load_model(self, model_id: str, 
                        optimization: str = "balanced") -> bool:
        """Load a model for use."""
        
        if model_id not in self.model_status:
            console.print(f"[red]Model {model_id} not found[/red]")
            return False
            
        model_info = self.model_status[model_id]
        
        # Download if needed
        if model_info.status == ModelStatus.NOT_DOWNLOADED:
            console.print(f"[yellow]Downloading {model_info.name} first...[/yellow]")
            if not await self.download_model(model_id):
                return False
                
        # Unload current model if different
        if self.current_model and self.current_model != model_id:
            self.unload_current_model()
            
        # Load the model
        success = await self.loader.load_model(model_info, optimization)
        
        if success:
            self.current_model = model_id
            self._save_model_cache()
            
        return success
        
    def switch_model(self, model_id: str) -> bool:
        """Switch to a different loaded model."""
        
        if model_id not in self.loader.loaded_models:
            console.print(f"[red]Model {model_id} not loaded[/red]")
            return False
            
        self.current_model = model_id
        console.print(f"[green]Switched to {model_id}[/green]")
        return True
        
    def unload_current_model(self) -> bool:
        """Unload the current model."""
        
        if not self.current_model:
            return True
            
        success = self.loader.unload_model(self.current_model)
        
        if success:
            # Update status
            if self.current_model in self.model_status:
                self.model_status[self.current_model].status = ModelStatus.DOWNLOADED
                self.model_status[self.current_model].memory_usage_mb = None
                
            self.current_model = None
            console.print("[green]Model unloaded[/green]")
            
        return success
        
    def get_recommendations(self) -> List[ModelInfo]:
        """Get model recommendations for current system."""
        
        # Get system memory
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Get recommendations
        recommendations = ModelRegistry.recommend_models(
            available_memory_gb, self.loader.device
        )
        
        return recommendations
        
    def benchmark_model(self, model_id: str, test_prompts: List[str]) -> Dict[str, Any]:
        """Benchmark a model's performance."""
        
        if model_id not in self.loader.loaded_pipelines:
            console.print(f"[red]Model {model_id} not loaded[/red]")
            return {}
            
        pipeline = self.loader.loaded_pipelines[model_id]
        
        results = {
            "model_id": model_id,
            "timestamp": time.time(),
            "prompts_tested": len(test_prompts),
            "inference_times": [],
            "tokens_per_second": [],
            "average_inference_time": 0.0,
            "average_tokens_per_second": 0.0
        }
        
        console.print(f"[blue]Benchmarking {model_id}...[/blue]")
        
        for prompt in test_prompts:
            start_time = time.time()
            
            response = pipeline(
                prompt,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Estimate tokens (rough approximation)
            estimated_tokens = len(response[0]["generated_text"].split()) * 1.3
            tokens_per_second = estimated_tokens / inference_time
            
            results["inference_times"].append(inference_time)
            results["tokens_per_second"].append(tokens_per_second)
            
        # Calculate averages
        results["average_inference_time"] = sum(results["inference_times"]) / len(results["inference_times"])
        results["average_tokens_per_second"] = sum(results["tokens_per_second"]) / len(results["tokens_per_second"])
        
        # Store in history
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        self.performance_history[model_id].append(results)
        
        return results
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system and model status."""
        
        import psutil
        
        # System info
        memory = psutil.virtual_memory()
        
        status = {
            "system": {
                "device": self.loader.device,
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_percent": memory.percent,
                "torch_version": torch.__version__,
                "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
                "cuda_available": torch.cuda.is_available()
            },
            "models": {
                "total_available": len(ModelRegistry.MODELS),
                "downloaded": len([m for m in self.model_status.values() 
                                if m.status in [ModelStatus.DOWNLOADED, ModelStatus.LOADED]]),
                "loaded": len(self.loader.loaded_models),
                "current_model": self.current_model,
                "cache_size_mb": self._calculate_cache_size()
            },
            "recommendations": [m.model_id for m in self.get_recommendations()[:3]]
        }
        
        return status
        
    def cleanup_cache(self, keep_current: bool = True) -> Dict[str, Any]:
        """Clean up model cache."""
        
        deleted_models = []
        freed_space_mb = 0
        
        for model_id, model_info in self.model_status.items():
            # Skip current model if requested
            if keep_current and model_id == self.current_model:
                continue
                
            # Skip loaded models
            if model_info.status == ModelStatus.LOADED:
                continue
                
            # Delete downloaded but unused models
            if (model_info.status == ModelStatus.DOWNLOADED and 
                model_info.local_path and model_info.local_path.exists()):
                
                size_mb = model_info.disk_size_mb or 0
                
                if self.downloader.delete_model(model_info):
                    deleted_models.append(model_id)
                    freed_space_mb += size_mb
                    
        # Update cache
        self._save_model_cache()
        
        return {
            "deleted_models": deleted_models,
            "freed_space_mb": freed_space_mb,
            "remaining_models": len(self.model_status) - len(deleted_models)
        }
        
    def _calculate_cache_size(self) -> int:
        """Calculate total cache size in MB."""
        
        total_size = 0
        
        for model_info in self.model_status.values():
            if model_info.disk_size_mb:
                total_size += model_info.disk_size_mb
                
        return total_size
        
    def _load_model_cache(self):
        """Load cached model information."""
        
        cache_file = self.cache_dir / "model_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                for model_id, data in cache_data.items():
                    if model_id in ModelRegistry.MODELS:
                        model_info = ModelRegistry.MODELS[model_id]
                        
                        # Update with cached data
                        model_info.disk_size_mb = data.get("disk_size_mb")
                        model_info.last_used = data.get("last_used")
                        model_info.performance_score = data.get("performance_score")
                        
                        # Check if files still exist
                        local_path = data.get("local_path")
                        if local_path and Path(local_path).exists():
                            model_info.local_path = Path(local_path)
                            model_info.status = ModelStatus.DOWNLOADED
                        else:
                            model_info.status = ModelStatus.NOT_DOWNLOADED
                            
                        self.model_status[model_id] = model_info
                        
            except Exception as e:
                logger.warning(f"Failed to load model cache: {e}")
                
    def _save_model_cache(self):
        """Save model information to cache."""
        
        cache_file = self.cache_dir / "model_cache.json"
        cache_data = {}
        
        for model_id, model_info in self.model_status.items():
            cache_data[model_id] = {
                "local_path": str(model_info.local_path) if model_info.local_path else None,
                "disk_size_mb": model_info.disk_size_mb,
                "memory_usage_mb": model_info.memory_usage_mb,
                "last_used": model_info.last_used,
                "performance_score": model_info.performance_score,
                "status": model_info.status.value
            }
            
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")


async def main():
    """Example usage of Model Manager."""
    
    manager = ModelManager()
    
    console.print("[blue]Saorsa Model Manager Demo[/blue]\n")
    
    # Show available models
    table = manager.list_available_models()
    console.print(table)
    
    # Show system status
    status = manager.get_system_status()
    console.print(f"\n[blue]System Status:[/blue]")
    console.print(f"Device: {status['system']['device']}")
    console.print(f"Available Memory: {status['system']['memory_available_gb']:.1f}GB")
    console.print(f"Models Downloaded: {status['models']['downloaded']}")
    console.print(f"Models Loaded: {status['models']['loaded']}")
    
    # Show recommendations
    recommendations = manager.get_recommendations()
    console.print(f"\n[blue]Recommended Models:[/blue]")
    for model in recommendations[:3]:
        console.print(f"  • {model.name} ({model.size.value})")
        
    # Test downloading a small model
    console.print(f"\n[blue]Testing model download...[/blue]")
    success = await manager.download_model("HuggingFaceTB/SmolLM2-135M-Instruct")
    
    if success:
        console.print(f"[green]✓ Download successful[/green]")
        
        # Test loading
        console.print(f"\n[blue]Testing model loading...[/blue]")
        success = await manager.load_model("HuggingFaceTB/SmolLM2-135M-Instruct")
        
        if success:
            console.print(f"[green]✓ Model loaded successfully[/green]")
            
            # Test benchmarking
            test_prompts = [
                "Move the robot arm to the left",
                "Pick up the red block",
                "Place the object on the table"
            ]
            
            console.print(f"\n[blue]Running benchmark...[/blue]")
            results = manager.benchmark_model(
                "HuggingFaceTB/SmolLM2-135M-Instruct", 
                test_prompts
            )
            
            console.print(f"Average inference time: {results['average_inference_time']:.3f}s")
            console.print(f"Average tokens/second: {results['average_tokens_per_second']:.1f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())