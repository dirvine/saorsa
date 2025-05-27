#!/usr/bin/env python3
"""
Model Download Script for Saorse (Updated for Phase 2)

This script downloads and sets up the AI models required for Saorse:
- OpenAI Whisper models for speech recognition
- Hugging Face local models (SmolLM2, Qwen2.5, Phi-3, Gemma-2)
"""

import os
import sys
import argparse
import requests
import hashlib
from pathlib import Path
from typing import Dict, Optional, List
import json
import time

try:
    import whisper
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.table import Table
    from huggingface_hub import snapshot_download, list_repo_files
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

console = Console()

# Model configurations (Updated for Phase 2)
WHISPER_MODELS = {
    "tiny": {
        "size": "39MB",
        "speed": "Very Fast",
        "accuracy": "Low",
        "description": "Fastest model, good for testing"
    },
    "base": {
        "size": "142MB", 
        "speed": "Fast",
        "accuracy": "Good",
        "description": "Recommended for most users"
    },
    "small": {
        "size": "483MB",
        "speed": "Medium", 
        "accuracy": "Better",
        "description": "Better accuracy, still efficient on M3"
    },
    "medium": {
        "size": "1.5GB",
        "speed": "Slow",
        "accuracy": "High",
        "description": "High accuracy, requires more resources"
    },
    "large": {
        "size": "3.1GB",
        "speed": "Very Slow", 
        "accuracy": "Highest",
        "description": "Best accuracy, only for powerful systems"
    }
}

# Phase 2: Local Hugging Face Models
HUGGINGFACE_MODELS = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": {
        "name": "SmolLM2-135M",
        "size": "500MB",
        "parameters": "135M",
        "speed": "Very Fast",
        "description": "Ultra-compact model, ideal for real-time robotics",
        "recommended_for": ["speed", "memory-constrained", "real-time"]
    },
    "HuggingFaceTB/SmolLM2-360M-Instruct": {
        "name": "SmolLM2-360M",
        "size": "1.2GB", 
        "parameters": "360M",
        "speed": "Fast",
        "description": "Balanced model, recommended for most use cases",
        "recommended_for": ["balanced", "general-purpose", "robotics"]
    },
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "name": "SmolLM2-1.7B",
        "size": "3.2GB",
        "parameters": "1.7B", 
        "speed": "Medium",
        "description": "Larger model with enhanced reasoning capabilities",
        "recommended_for": ["quality", "complex-reasoning", "advanced-tasks"]
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "name": "Qwen2.5-1.5B",
        "size": "2.8GB",
        "parameters": "1.5B",
        "speed": "Medium",
        "description": "Multilingual model with strong reasoning",
        "recommended_for": ["reasoning", "multilingual", "complex-commands"]
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "name": "Phi-3-Mini",
        "size": "7.6GB",
        "parameters": "3.8B",
        "speed": "Slow",
        "description": "Microsoft's high-performance compact model",
        "recommended_for": ["quality", "reasoning", "high-memory"]
    },
    "google/gemma-2-2b-it": {
        "name": "Gemma-2-2B",
        "size": "4.2GB", 
        "parameters": "2B",
        "speed": "Medium",
        "description": "Google's efficient instruction-tuned model",
        "recommended_for": ["balanced", "instruction-following", "general"]
    }
}

# Predefined model sets for different use cases
MODEL_SETS = {
    "minimal": {
        "whisper": ["tiny"],
        "huggingface": ["HuggingFaceTB/SmolLM2-135M-Instruct"],
        "description": "Minimal set for testing and development",
        "total_size": "~540MB"
    },
    "recommended": {
        "whisper": ["base"],
        "huggingface": ["HuggingFaceTB/SmolLM2-360M-Instruct"],
        "description": "Recommended set for most users",
        "total_size": "~1.4GB"
    },
    "advanced": {
        "whisper": ["small"],
        "huggingface": [
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        ],
        "description": "Advanced set with multiple model options",
        "total_size": "~7.0GB"
    },
    "research": {
        "whisper": ["base", "small"],
        "huggingface": [
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct"
        ],
        "description": "Research set with diverse model types",
        "total_size": "~12GB"
    }
}


def get_models_directory() -> Path:
    """Get the models directory path."""
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"
    
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        
    return models_dir


def check_disk_space(required_gb: float) -> bool:
    """Check if there's enough disk space."""
    models_dir = get_models_directory()
    stat = os.statvfs(models_dir)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    
    return free_gb >= required_gb


def get_model_size_gb(model_size_str: str) -> float:
    """Convert model size string to GB."""
    if "MB" in model_size_str:
        return float(model_size_str.replace("MB", "")) / 1024
    elif "GB" in model_size_str:
        return float(model_size_str.replace("GB", ""))
    return 0.0


def download_whisper_model(model_name: str, force: bool = False) -> bool:
    """Download a Whisper model."""
    models_dir = get_models_directory()
    
    # Check if model already exists
    model_file = models_dir / f"whisper-{model_name}.pt"
    if model_file.exists() and not force:
        console.print(f"[yellow]Whisper {model_name} model already exists[/yellow]")
        return True
        
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task(f"Downloading Whisper {model_name}...", total=100)
            
            console.print(f"[blue]Downloading Whisper {model_name} model...[/blue]")
            model = whisper.load_model(model_name, download_root=str(models_dir))
            
            if model:
                console.print(f"[green]✓ Whisper {model_name} model downloaded successfully[/green]")
                return True
            else:
                console.print(f"[red]✗ Failed to download Whisper {model_name} model[/red]")
                return False
                
    except Exception as e:
        console.print(f"[red]✗ Error downloading Whisper {model_name}: {e}[/red]")
        return False


def download_huggingface_model(model_id: str, force: bool = False) -> bool:
    """Download a Hugging Face model."""
    models_dir = get_models_directory()
    model_info = HUGGINGFACE_MODELS.get(model_id)
    
    if not model_info:
        console.print(f"[red]Unknown model: {model_id}[/red]")
        return False
        
    # Create local directory name
    local_name = model_id.replace("/", "_")
    local_path = models_dir / local_name
    
    if local_path.exists() and not force:
        console.print(f"[yellow]{model_info['name']} already exists[/yellow]")
        return True
        
    try:
        console.print(f"[blue]Downloading {model_info['name']} ({model_info['size']})...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            
            download_task = progress.add_task(
                f"Downloading {model_info['name']}...", 
                total=None
            )
            
            # Download model files
            downloaded_path = snapshot_download(
                model_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            progress.update(download_task, completed=True)
            
        # Test loading the model
        console.print(f"[yellow]Testing {model_info['name']}...[/yellow]")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
            # Quick test - don't fully load model to save time
            console.print(f"[green]✓ {model_info['name']} downloaded and verified[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Model verification failed: {e}[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]✗ Failed to download {model_info['name']}: {e}[/red]")
        return False


def test_whisper_model(model_name: str) -> bool:
    """Test a Whisper model by loading it."""
    try:
        models_dir = get_models_directory()
        console.print(f"[blue]Testing Whisper {model_name} model...[/blue]")
        
        model = whisper.load_model(model_name, download_root=str(models_dir))
        
        # Test with dummy audio
        import numpy as np
        dummy_audio = np.random.randn(16000).astype(np.float32)
        
        result = model.transcribe(dummy_audio)
        
        if result:
            console.print(f"[green]✓ Whisper {model_name} model test successful[/green]")
            return True
        else:
            console.print(f"[red]✗ Whisper {model_name} model test failed[/red]")
            return False
            
    except Exception as e:
        console.print(f"[red]✗ Error testing Whisper {model_name}: {e}[/red]")
        return False


def test_huggingface_model(model_id: str) -> bool:
    """Test a Hugging Face model by loading it."""
    try:
        models_dir = get_models_directory()
        local_name = model_id.replace("/", "_")
        local_path = models_dir / local_name
        
        if not local_path.exists():
            console.print(f"[red]Model {model_id} not found locally[/red]")
            return False
            
        model_info = HUGGINGFACE_MODELS.get(model_id, {})
        console.print(f"[blue]Testing {model_info.get('name', model_id)}...[/blue]")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
        
        # Load model (on CPU to avoid memory issues)
        model = AutoModelForCausalLM.from_pretrained(
            str(local_path), 
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Quick inference test
        test_prompt = "Move the robot arm to"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        console.print(f"[green]✓ {model_info.get('name', model_id)} test successful[/green]")
        console.print(f"[dim]Test output: {response}[/dim]")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Error testing {model_id}: {e}[/red]")
        return False


def list_available_models():
    """List all available models with details."""
    
    # Whisper models table
    table = Table(title="Available Whisper Models")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="magenta") 
    table.add_column("Speed", style="green")
    table.add_column("Accuracy", style="yellow")
    table.add_column("Description", style="dim")
    
    for name, info in WHISPER_MODELS.items():
        table.add_row(
            f"whisper-{name}",
            info["size"],
            info["speed"], 
            info["accuracy"],
            info["description"]
        )
        
    console.print(table)
    
    # Hugging Face models table  
    table2 = Table(title="Available Hugging Face Models (Phase 2)")
    table2.add_column("Model", style="cyan")
    table2.add_column("Parameters", style="magenta")
    table2.add_column("Size", style="red")
    table2.add_column("Speed", style="green")
    table2.add_column("Description", style="dim")
    
    for model_id, info in HUGGINGFACE_MODELS.items():
        table2.add_row(
            info["name"],
            info["parameters"],
            info["size"],
            info["speed"],
            info["description"]
        )
        
    console.print("\n")
    console.print(table2)
    
    # Model sets table
    table3 = Table(title="Predefined Model Sets")
    table3.add_column("Set", style="cyan")
    table3.add_column("Total Size", style="magenta")
    table3.add_column("Description", style="dim")
    
    for set_name, set_info in MODEL_SETS.items():
        table3.add_row(
            set_name,
            set_info["total_size"],
            set_info["description"]
        )
        
    console.print("\n")
    console.print(table3)


def check_installed_models():
    """Check which models are already installed."""
    models_dir = get_models_directory()
    
    table = Table(title="Installed Models")
    table.add_column("Model", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Size", style="magenta")
    table.add_column("Location", style="dim")
    
    # Check Whisper models
    for model_name in WHISPER_MODELS.keys():
        model_files = list(models_dir.glob(f"*{model_name}*"))
        if model_files:
            model_file = model_files[0]
            size_mb = model_file.stat().st_size / (1024*1024)
            table.add_row(
                f"whisper-{model_name}",
                "Whisper",
                "✓ Installed",
                f"{size_mb:.1f}MB",
                str(model_file)
            )
        else:
            table.add_row(
                f"whisper-{model_name}",
                "Whisper",
                "✗ Not installed",
                "N/A",
                "N/A"
            )
            
    # Check Hugging Face models
    for model_id, model_info in HUGGINGFACE_MODELS.items():
        local_name = model_id.replace("/", "_")
        local_path = models_dir / local_name
        
        if local_path.exists():
            # Calculate directory size
            total_size = sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024*1024)
            
            table.add_row(
                model_info["name"],
                "Hugging Face",
                "✓ Installed", 
                f"{size_mb:.1f}MB",
                str(local_path)
            )
        else:
            table.add_row(
                model_info["name"],
                "Hugging Face",
                "✗ Not installed",
                "N/A",
                "N/A"
            )
            
    console.print(table)


def get_recommended_models():
    """Get recommended models based on system."""
    import platform
    import psutil
    
    # Get system info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    is_apple_silicon = platform.machine() == "arm64" and platform.system() == "Darwin"
    
    if is_apple_silicon and memory_gb >= 32:
        return MODEL_SETS["advanced"]
    elif is_apple_silicon and memory_gb >= 16:
        return MODEL_SETS["recommended"]
    elif memory_gb >= 16:
        return MODEL_SETS["recommended"]
    else:
        return MODEL_SETS["minimal"]


def download_model_set(set_name: str, force: bool = False) -> bool:
    """Download a predefined model set."""
    
    if set_name not in MODEL_SETS:
        console.print(f"[red]Unknown model set: {set_name}[/red]")
        return False
        
    model_set = MODEL_SETS[set_name]
    
    console.print(f"[blue]Downloading '{set_name}' model set...[/blue]")
    console.print(f"[dim]{model_set['description']}[/dim]")
    console.print(f"[dim]Total size: {model_set['total_size']}[/dim]\n")
    
    success_count = 0
    total_models = len(model_set["whisper"]) + len(model_set["huggingface"])
    
    # Download Whisper models
    for model_name in model_set["whisper"]:
        if download_whisper_model(model_name, force):
            success_count += 1
            
    # Download Hugging Face models
    for model_id in model_set["huggingface"]:
        if download_huggingface_model(model_id, force):
            success_count += 1
            
    if success_count == total_models:
        console.print(f"\n[green]✓ All {total_models} models downloaded successfully![/green]")
        return True
    else:
        console.print(f"\n[yellow]⚠ {success_count}/{total_models} models downloaded[/yellow]")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download AI models for Saorse (Phase 2)")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--check", action="store_true", help="Check installed models")
    parser.add_argument("--whisper", choices=list(WHISPER_MODELS.keys()), 
                       help="Download specific Whisper model")
    parser.add_argument("--hf-model", choices=list(HUGGINGFACE_MODELS.keys()),
                       help="Download specific Hugging Face model")
    parser.add_argument("--set", choices=list(MODEL_SETS.keys()),
                       help="Download predefined model set")
    parser.add_argument("--recommended", action="store_true", 
                       help="Download recommended models for your system")
    parser.add_argument("--all-whisper", action="store_true", help="Download all Whisper models")
    parser.add_argument("--all-hf", action="store_true", help="Download all Hugging Face models")
    parser.add_argument("--force", action="store_true", help="Force redownload existing models")
    parser.add_argument("--test", action="store_true", help="Test downloaded models")
    
    args = parser.parse_args()
    
    # Show banner
    console.print(Panel.fit(
        "[bold blue]Saorse Model Downloader (Phase 2)[/bold blue]\n"
        "[dim]Download local AI models for voice-controlled robotics[/dim]",
        title="🤖 AI Model Management",
        border_style="blue"
    ))
    
    if args.list:
        list_available_models()
        return
        
    if args.check:
        check_installed_models()
        return
        
    # Determine what to download
    models_to_download = {"whisper": [], "huggingface": []}
    
    if args.whisper:
        models_to_download["whisper"].append(args.whisper)
    elif args.all_whisper:
        models_to_download["whisper"] = list(WHISPER_MODELS.keys())
        
    if args.hf_model:
        models_to_download["huggingface"].append(args.hf_model)
    elif args.all_hf:
        models_to_download["huggingface"] = list(HUGGINGFACE_MODELS.keys())
        
    if args.set:
        success = download_model_set(args.set, args.force)
        if success and args.test:
            console.print("\n[blue]Testing downloaded models...[/blue]")
            # Test models from the set
            set_info = MODEL_SETS[args.set]
            for model_name in set_info["whisper"]:
                test_whisper_model(model_name)
            for model_id in set_info["huggingface"]:
                test_huggingface_model(model_id)
        return
        
    if args.recommended:
        recommended = get_recommended_models()
        console.print(f"[blue]Downloading recommended models for your system...[/blue]")
        success = download_model_set("recommended", args.force)
        if success and args.test:
            console.print("\n[blue]Testing downloaded models...[/blue]")
            for model_name in recommended["whisper"]:
                test_whisper_model(model_name)
            for model_id in recommended["huggingface"]:
                test_huggingface_model(model_id)
        return
        
    # Interactive mode if no specific models chosen
    if not any([models_to_download["whisper"], models_to_download["huggingface"]]):
        console.print("[yellow]No models specified. Showing recommendations...[/yellow]")
        recommended = get_recommended_models()
        console.print(f"[blue]Recommended model set: '{recommended.get('name', 'recommended')}'[/blue]")
        console.print(f"[dim]{recommended['description']}[/dim]")
        
        response = console.input("Download recommended models? [Y/n]: ").lower()
        if response in ['', 'y', 'yes']:
            download_model_set("recommended", args.force)
        else:
            console.print("Use --help to see all options")
        return
        
    # Download specified models
    if models_to_download["whisper"] or models_to_download["huggingface"]:
        total_models = len(models_to_download["whisper"]) + len(models_to_download["huggingface"])
        console.print(f"[blue]Downloading {total_models} models...[/blue]")
        
        success_count = 0
        
        # Download Whisper models
        for model_name in models_to_download["whisper"]:
            if download_whisper_model(model_name, args.force):
                success_count += 1
                if args.test:
                    test_whisper_model(model_name)
                    
        # Download Hugging Face models
        for model_id in models_to_download["huggingface"]:
            if download_huggingface_model(model_id, args.force):
                success_count += 1
                if args.test:
                    test_huggingface_model(model_id)
                    
        # Summary
        if success_count == total_models:
            console.print(f"[green]✓ All {success_count} models downloaded successfully![/green]")
        else:
            console.print(f"[yellow]⚠ {success_count}/{total_models} models downloaded[/yellow]")
            
    # Show what's installed
    console.print("\n[blue]Current model status:[/blue]")
    check_installed_models()
    
    # Usage tips
    console.print("\n[green]Next steps:[/green]")
    console.print("1. Test audio: python src/main_mac.py test-audio")
    console.print("2. Connect robot and run: ./launch.sh <port>")
    console.print("3. Try advanced voice commands with AI models")


if __name__ == "__main__":
    main()