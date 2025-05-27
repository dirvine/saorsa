#!/usr/bin/env python3
"""
Performance Monitor for Saorse Robot System

This module provides comprehensive performance monitoring including system metrics,
AI model inference times, audio processing latency, and robot response times.
"""

import asyncio
import logging
import threading
import time
import psutil
import platform
from typing import Dict, List, Optional, Callable, Any, Deque
from dataclasses import dataclass, field
from collections import deque
import json

import torch
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource usage metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    temperature: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None


@dataclass
class AIMetrics:
    """AI model performance metrics."""
    timestamp: float
    model_name: str
    inference_time_ms: float
    input_size: int
    output_size: Optional[int] = None
    device: str = "cpu"
    memory_used_mb: Optional[float] = None


@dataclass
class AudioMetrics:
    """Audio processing performance metrics."""
    timestamp: float
    processing_time_ms: float
    audio_duration_ms: float
    sample_rate: int
    chunk_size: int
    vad_time_ms: Optional[float] = None
    transcription_time_ms: Optional[float] = None


@dataclass
class RobotMetrics:
    """Robot control performance metrics."""
    timestamp: float
    command_to_execution_ms: float
    movement_completion_ms: float
    communication_latency_ms: float
    motor_count: int
    success_rate: float


@dataclass
class PerformanceThresholds:
    """Performance warning thresholds."""
    cpu_warning: float = 80.0  # percent
    cpu_critical: float = 95.0  # percent
    memory_warning: float = 80.0  # percent
    memory_critical: float = 95.0  # percent
    inference_warning_ms: float = 200.0  # milliseconds
    inference_critical_ms: float = 500.0  # milliseconds
    audio_latency_warning_ms: float = 100.0  # milliseconds
    audio_latency_critical_ms: float = 300.0  # milliseconds
    robot_response_warning_ms: float = 200.0  # milliseconds
    robot_response_critical_ms: float = 1000.0  # milliseconds
    temperature_warning: float = 70.0  # Celsius
    temperature_critical: float = 85.0  # Celsius


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Tracks system resources, AI model performance, audio processing,
    and robot control metrics with alerting and reporting capabilities.
    """
    
    def __init__(self, thresholds: Optional[PerformanceThresholds] = None):
        self.thresholds = thresholds or PerformanceThresholds()
        
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Metric storage (last 1000 samples each)
        self.system_metrics: Deque[SystemMetrics] = deque(maxlen=1000)
        self.ai_metrics: Deque[AIMetrics] = deque(maxlen=1000)
        self.audio_metrics: Deque[AudioMetrics] = deque(maxlen=1000)
        self.robot_metrics: Deque[RobotMetrics] = deque(maxlen=1000)
        
        # Performance callbacks
        self.warning_callbacks: List[Callable[[str, str], None]] = []
        
        # System info
        self.system_info = self._get_system_info()
        
        # Timing helpers
        self._timing_stack: List[Tuple[str, float]] = []
        
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start performance monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.interval = interval
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        console.print("[green]📊 Performance monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
            
        console.print("[yellow]📊 Performance monitoring stopped")
        
    def add_warning_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add callback for performance warnings."""
        self.warning_callbacks.append(callback)
        
    def record_ai_inference(self, model_name: str, inference_time: float, 
                           input_size: int, device: str = "cpu", 
                           output_size: Optional[int] = None) -> None:
        """Record AI model inference metrics."""
        metrics = AIMetrics(
            timestamp=time.time(),
            model_name=model_name,
            inference_time_ms=inference_time * 1000,
            input_size=input_size,
            output_size=output_size,
            device=device,
            memory_used_mb=self._get_gpu_memory_usage() if device.startswith("cuda") or device == "mps" else None
        )
        
        self.ai_metrics.append(metrics)
        
        # Check thresholds
        if metrics.inference_time_ms > self.thresholds.inference_critical_ms:
            self._trigger_warning("AI_INFERENCE", 
                f"Critical: {model_name} inference time {metrics.inference_time_ms:.1f}ms")
        elif metrics.inference_time_ms > self.thresholds.inference_warning_ms:
            self._trigger_warning("AI_INFERENCE", 
                f"Warning: {model_name} inference time {metrics.inference_time_ms:.1f}ms")
                
    def record_audio_processing(self, processing_time: float, audio_duration: float,
                               sample_rate: int, chunk_size: int,
                               vad_time: Optional[float] = None,
                               transcription_time: Optional[float] = None) -> None:
        """Record audio processing metrics."""
        metrics = AudioMetrics(
            timestamp=time.time(),
            processing_time_ms=processing_time * 1000,
            audio_duration_ms=audio_duration * 1000,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            vad_time_ms=vad_time * 1000 if vad_time else None,
            transcription_time_ms=transcription_time * 1000 if transcription_time else None
        )
        
        self.audio_metrics.append(metrics)
        
        # Check latency thresholds
        if metrics.processing_time_ms > self.thresholds.audio_latency_critical_ms:
            self._trigger_warning("AUDIO_LATENCY", 
                f"Critical: Audio processing latency {metrics.processing_time_ms:.1f}ms")
        elif metrics.processing_time_ms > self.thresholds.audio_latency_warning_ms:
            self._trigger_warning("AUDIO_LATENCY", 
                f"Warning: Audio processing latency {metrics.processing_time_ms:.1f}ms")
                
    def record_robot_command(self, command_to_execution: float, 
                           movement_completion: float, communication_latency: float,
                           motor_count: int, success: bool) -> None:
        """Record robot control metrics."""
        # Update success rate (simple moving average)
        recent_robot_metrics = list(self.robot_metrics)[-20:]  # Last 20 commands
        if recent_robot_metrics:
            recent_successes = sum(1 for m in recent_robot_metrics if m.success_rate > 0.5)
            success_rate = (recent_successes + (1 if success else 0)) / (len(recent_robot_metrics) + 1)
        else:
            success_rate = 1.0 if success else 0.0
            
        metrics = RobotMetrics(
            timestamp=time.time(),
            command_to_execution_ms=command_to_execution * 1000,
            movement_completion_ms=movement_completion * 1000,
            communication_latency_ms=communication_latency * 1000,
            motor_count=motor_count,
            success_rate=success_rate
        )
        
        self.robot_metrics.append(metrics)
        
        # Check response time thresholds
        total_response = metrics.command_to_execution_ms + metrics.movement_completion_ms
        if total_response > self.thresholds.robot_response_critical_ms:
            self._trigger_warning("ROBOT_RESPONSE", 
                f"Critical: Robot response time {total_response:.1f}ms")
        elif total_response > self.thresholds.robot_response_warning_ms:
            self._trigger_warning("ROBOT_RESPONSE", 
                f"Warning: Robot response time {total_response:.1f}ms")
                
        # Check success rate
        if success_rate < 0.8:
            self._trigger_warning("ROBOT_RELIABILITY", 
                f"Warning: Robot success rate {success_rate:.1%}")
                
    def start_timing(self, operation_name: str) -> None:
        """Start timing an operation."""
        self._timing_stack.append((operation_name, time.time()))
        
    def end_timing(self, operation_name: str) -> float:
        """End timing an operation and return duration."""
        if not self._timing_stack:
            logger.warning(f"No timing started for {operation_name}")
            return 0.0
            
        stack_name, start_time = self._timing_stack.pop()
        if stack_name != operation_name:
            logger.warning(f"Timing mismatch: expected {stack_name}, got {operation_name}")
            
        return time.time() - start_time
        
    def get_current_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_gb = (memory.total - memory.available) / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        # Temperature (macOS specific)
        temperature = self._get_cpu_temperature()
        
        # GPU memory (if available)
        gpu_memory_used, gpu_memory_total = self._get_gpu_memory_info()
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            disk_usage_percent=disk_usage_percent,
            temperature=temperature,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total
        )
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_time = time.time()
        
        # System metrics summary
        system_summary = {}
        if self.system_metrics:
            recent_system = [m for m in self.system_metrics if current_time - m.timestamp < 60]
            if recent_system:
                system_summary = {
                    "avg_cpu_percent": np.mean([m.cpu_percent for m in recent_system]),
                    "max_cpu_percent": np.max([m.cpu_percent for m in recent_system]),
                    "avg_memory_percent": np.mean([m.memory_percent for m in recent_system]),
                    "max_memory_percent": np.max([m.memory_percent for m in recent_system]),
                    "current_memory_gb": recent_system[-1].memory_used_gb,
                    "avg_temperature": np.mean([m.temperature for m in recent_system if m.temperature]),
                    "max_temperature": np.max([m.temperature for m in recent_system if m.temperature]) if any(m.temperature for m in recent_system) else None
                }
                
        # AI metrics summary
        ai_summary = {}
        if self.ai_metrics:
            recent_ai = [m for m in self.ai_metrics if current_time - m.timestamp < 300]  # 5 minutes
            if recent_ai:
                by_model = {}
                for metric in recent_ai:
                    if metric.model_name not in by_model:
                        by_model[metric.model_name] = []
                    by_model[metric.model_name].append(metric.inference_time_ms)
                    
                ai_summary = {
                    "models": {
                        model: {
                            "avg_inference_ms": np.mean(times),
                            "max_inference_ms": np.max(times),
                            "count": len(times)
                        }
                        for model, times in by_model.items()
                    },
                    "total_inferences": len(recent_ai)
                }
                
        # Audio metrics summary
        audio_summary = {}
        if self.audio_metrics:
            recent_audio = [m for m in self.audio_metrics if current_time - m.timestamp < 300]
            if recent_audio:
                audio_summary = {
                    "avg_processing_ms": np.mean([m.processing_time_ms for m in recent_audio]),
                    "max_processing_ms": np.max([m.processing_time_ms for m in recent_audio]),
                    "avg_realtime_factor": np.mean([m.audio_duration_ms / m.processing_time_ms for m in recent_audio]),
                    "total_processed": len(recent_audio)
                }
                
        # Robot metrics summary
        robot_summary = {}
        if self.robot_metrics:
            recent_robot = [m for m in self.robot_metrics if current_time - m.timestamp < 300]
            if recent_robot:
                robot_summary = {
                    "avg_response_ms": np.mean([m.command_to_execution_ms + m.movement_completion_ms for m in recent_robot]),
                    "max_response_ms": np.max([m.command_to_execution_ms + m.movement_completion_ms for m in recent_robot]),
                    "avg_success_rate": np.mean([m.success_rate for m in recent_robot]),
                    "total_commands": len(recent_robot)
                }
                
        return {
            "timestamp": current_time,
            "system": system_summary,
            "ai": ai_summary,
            "audio": audio_summary,
            "robot": robot_summary,
            "system_info": self.system_info
        }
        
    def print_performance_table(self) -> None:
        """Print formatted performance table to console."""
        summary = self.get_performance_summary()
        
        table = Table(title="Saorse Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")
        
        # System metrics
        if summary["system"]:
            s = summary["system"]
            cpu_status = "🔴 Critical" if s.get("max_cpu_percent", 0) > self.thresholds.cpu_critical else "🟡 Warning" if s.get("max_cpu_percent", 0) > self.thresholds.cpu_warning else "🟢 Good"
            table.add_row("CPU Usage", f"{s.get('avg_cpu_percent', 0):.1f}% (max: {s.get('max_cpu_percent', 0):.1f}%)", cpu_status)
            
            mem_status = "🔴 Critical" if s.get("max_memory_percent", 0) > self.thresholds.memory_critical else "🟡 Warning" if s.get("max_memory_percent", 0) > self.thresholds.memory_warning else "🟢 Good"
            table.add_row("Memory Usage", f"{s.get('current_memory_gb', 0):.1f}GB ({s.get('avg_memory_percent', 0):.1f}%)", mem_status)
            
            if s.get("max_temperature"):
                temp_status = "🔴 Critical" if s["max_temperature"] > self.thresholds.temperature_critical else "🟡 Warning" if s["max_temperature"] > self.thresholds.temperature_warning else "🟢 Good"
                table.add_row("Temperature", f"{s.get('avg_temperature', 0):.1f}°C (max: {s['max_temperature']:.1f}°C)", temp_status)
                
        # AI metrics
        if summary["ai"]:
            ai = summary["ai"]
            table.add_row("Total AI Inferences", str(ai["total_inferences"]), "")
            
            for model, stats in ai["models"].items():
                inference_status = "🔴 Critical" if stats["max_inference_ms"] > self.thresholds.inference_critical_ms else "🟡 Warning" if stats["max_inference_ms"] > self.thresholds.inference_warning_ms else "🟢 Good"
                table.add_row(f"{model} Inference", f"{stats['avg_inference_ms']:.1f}ms (max: {stats['max_inference_ms']:.1f}ms)", inference_status)
                
        # Audio metrics
        if summary["audio"]:
            audio = summary["audio"]
            audio_status = "🔴 Critical" if audio["max_processing_ms"] > self.thresholds.audio_latency_critical_ms else "🟡 Warning" if audio["max_processing_ms"] > self.thresholds.audio_latency_warning_ms else "🟢 Good"
            table.add_row("Audio Processing", f"{audio['avg_processing_ms']:.1f}ms (realtime: {audio['avg_realtime_factor']:.1f}x)", audio_status)
            
        # Robot metrics
        if summary["robot"]:
            robot = summary["robot"]
            robot_status = "🔴 Critical" if robot["max_response_ms"] > self.thresholds.robot_response_critical_ms else "🟡 Warning" if robot["max_response_ms"] > self.thresholds.robot_response_warning_ms else "🟢 Good"
            table.add_row("Robot Response", f"{robot['avg_response_ms']:.1f}ms (success: {robot['avg_success_rate']:.1%})", robot_status)
            
        console.print(table)
        
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        data = {
            "system_metrics": [
                {
                    "timestamp": m.timestamp,
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "memory_used_gb": m.memory_used_gb,
                    "temperature": m.temperature
                }
                for m in self.system_metrics
            ],
            "ai_metrics": [
                {
                    "timestamp": m.timestamp,
                    "model_name": m.model_name,
                    "inference_time_ms": m.inference_time_ms,
                    "device": m.device
                }
                for m in self.ai_metrics
            ],
            "audio_metrics": [
                {
                    "timestamp": m.timestamp,
                    "processing_time_ms": m.processing_time_ms,
                    "audio_duration_ms": m.audio_duration_ms
                }
                for m in self.audio_metrics
            ],
            "robot_metrics": [
                {
                    "timestamp": m.timestamp,
                    "command_to_execution_ms": m.command_to_execution_ms,
                    "movement_completion_ms": m.movement_completion_ms,
                    "success_rate": m.success_rate
                }
                for m in self.robot_metrics
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        console.print(f"[green]📊 Metrics exported to {filepath}")
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__ if torch else None,
            "torch_mps_available": torch.backends.mps.is_available() if torch else False,
            "torch_cuda_available": torch.cuda.is_available() if torch else False
        }
        
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature (macOS specific)."""
        try:
            if platform.system() == "Darwin":
                # This is a simplified approach - real implementation would use
                # macOS specific APIs or external tools like 'sensors'
                return None
        except Exception:
            pass
        return None
        
    def _get_gpu_memory_info(self) -> Tuple[Optional[float], Optional[float]]:
        """Get GPU memory information."""
        try:
            if torch and torch.backends.mps.is_available():
                # MPS memory info is limited
                return None, None
            elif torch and torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                return memory_used, memory_total
        except Exception:
            pass
        return None, None
        
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        try:
            if torch and torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**2)  # MB
        except Exception:
            pass
        return None
        
    def _trigger_warning(self, category: str, message: str) -> None:
        """Trigger performance warning."""
        for callback in self.warning_callbacks:
            try:
                callback(category, message)
            except Exception as e:
                logger.error(f"Warning callback error: {e}")
                
        logger.warning(f"Performance {category}: {message}")
        
    def _monitoring_loop(self) -> None:
        """Main performance monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self.get_current_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Check system thresholds
                if system_metrics.cpu_percent > self.thresholds.cpu_critical:
                    self._trigger_warning("SYSTEM_CPU", 
                        f"Critical: CPU usage {system_metrics.cpu_percent:.1f}%")
                elif system_metrics.cpu_percent > self.thresholds.cpu_warning:
                    self._trigger_warning("SYSTEM_CPU", 
                        f"Warning: CPU usage {system_metrics.cpu_percent:.1f}%")
                        
                if system_metrics.memory_percent > self.thresholds.memory_critical:
                    self._trigger_warning("SYSTEM_MEMORY", 
                        f"Critical: Memory usage {system_metrics.memory_percent:.1f}%")
                elif system_metrics.memory_percent > self.thresholds.memory_warning:
                    self._trigger_warning("SYSTEM_MEMORY", 
                        f"Warning: Memory usage {system_metrics.memory_percent:.1f}%")
                        
                if (system_metrics.temperature and 
                    system_metrics.temperature > self.thresholds.temperature_critical):
                    self._trigger_warning("SYSTEM_TEMPERATURE", 
                        f"Critical: Temperature {system_metrics.temperature:.1f}°C")
                        
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(1.0)


async def main():
    """Example usage of PerformanceMonitor."""
    
    def warning_callback(category: str, message: str):
        console.print(f"[red]PERFORMANCE WARNING [{category}]: {message}")
        
    # Create monitor
    monitor = PerformanceMonitor()
    monitor.add_warning_callback(warning_callback)
    
    # Start monitoring
    monitor.start_monitoring(interval=0.5)
    
    # Simulate some operations
    await asyncio.sleep(2)
    
    # Simulate AI inference
    monitor.start_timing("whisper_inference")
    await asyncio.sleep(0.15)  # Simulate 150ms inference
    inference_time = monitor.end_timing("whisper_inference")
    monitor.record_ai_inference("whisper-base", inference_time, 1600, "mps")
    
    # Simulate audio processing
    monitor.record_audio_processing(0.08, 0.5, 16000, 8000)
    
    # Simulate robot command
    monitor.record_robot_command(0.05, 0.3, 0.02, 6, True)
    
    await asyncio.sleep(3)
    
    # Print performance table
    monitor.print_performance_table()
    
    # Export metrics
    monitor.export_metrics("/tmp/saorse_metrics.json")
    
    # Stop monitoring
    monitor.stop_monitoring()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())