#!/usr/bin/env python3
"""
Voice System Test Script for Saorse

This script provides comprehensive testing of the voice recognition system
including microphone input, Whisper model performance, and command processing.
"""

import asyncio
import sys
import time
import logging
from pathlib import Path
import argparse

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from mac_audio_handler import MacAudioHandler, AudioConfig
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    import sounddevice as sd
    import numpy as np
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you've activated the virtual environment and installed dependencies")
    sys.exit(1)

console = Console()


class VoiceSystemTester:
    """Comprehensive voice system testing."""
    
    def __init__(self):
        self.test_results = {}
        self.audio_handler = None
        
    def test_audio_devices(self) -> bool:
        """Test available audio input devices."""
        console.print("[blue]Testing audio devices...[/blue]")
        
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
                    
            if not input_devices:
                console.print("[red]✗ No audio input devices found[/red]")
                return False
                
            # Display available devices
            table = Table(title="Available Audio Input Devices")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Channels", style="yellow")
            table.add_column("Sample Rate", style="magenta")
            
            for device in input_devices:
                table.add_row(
                    str(device['id']),
                    device['name'],
                    str(device['channels']),
                    f"{device['sample_rate']:.0f} Hz"
                )
                
            console.print(table)
            
            # Test default device
            default_device = sd.default.device[0]
            console.print(f"[green]✓ Default input device: {devices[default_device]['name']}[/green]")
            
            self.test_results['audio_devices'] = True
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Audio device test failed: {e}[/red]")
            self.test_results['audio_devices'] = False
            return False
            
    def test_microphone_input(self, duration: float = 3.0) -> bool:
        """Test microphone input level."""
        console.print(f"[blue]Testing microphone input for {duration} seconds...[/blue]")
        console.print("[yellow]Please speak into your microphone[/yellow]")
        
        try:
            # Record audio
            sample_rate = 16000
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            # Analyze audio levels
            max_level = np.max(np.abs(audio_data))
            rms_level = np.sqrt(np.mean(audio_data**2))
            
            console.print(f"[blue]Audio Analysis:[/blue]")
            console.print(f"  Maximum level: {max_level:.4f}")
            console.print(f"  RMS level: {rms_level:.4f}")
            
            if max_level < 0.001:
                console.print("[red]✗ Very low audio level - check microphone connection[/red]")
                self.test_results['microphone_input'] = False
                return False
            elif max_level < 0.01:
                console.print("[yellow]⚠ Low audio level - speak louder or adjust microphone[/yellow]")
            else:
                console.print("[green]✓ Good audio level detected[/green]")
                
            self.test_results['microphone_input'] = True
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Microphone test failed: {e}[/red]")
            self.test_results['microphone_input'] = False
            return False
            
    def test_whisper_models(self) -> bool:
        """Test Whisper model loading and inference."""
        console.print("[blue]Testing Whisper models...[/blue]")
        
        try:
            import whisper
            import torch
            
            # Check device availability
            if torch.backends.mps.is_available():
                device = "mps"
                console.print("[green]✓ MPS (Metal Performance Shaders) available[/green]")
            elif torch.cuda.is_available():
                device = "cuda"
                console.print("[green]✓ CUDA available[/green]")
            else:
                device = "cpu"
                console.print("[yellow]⚠ Using CPU (consider upgrading for better performance)[/yellow]")
                
            # Test model loading
            models_to_test = ["tiny", "base"]
            
            for model_name in models_to_test:
                try:
                    console.print(f"[blue]Testing Whisper {model_name} model...[/blue]")
                    
                    start_time = time.time()
                    model = whisper.load_model(model_name, device=device)
                    load_time = time.time() - start_time
                    
                    console.print(f"  ✓ Model loaded in {load_time:.2f}s")
                    
                    # Test inference with dummy audio
                    dummy_audio = np.random.randn(16000).astype(np.float32)
                    
                    start_time = time.time()
                    result = model.transcribe(dummy_audio)
                    inference_time = time.time() - start_time
                    
                    console.print(f"  ✓ Inference completed in {inference_time:.2f}s")
                    
                except Exception as e:
                    console.print(f"  ✗ {model_name} model test failed: {e}")
                    
            self.test_results['whisper_models'] = True
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Whisper model test failed: {e}[/red]")
            self.test_results['whisper_models'] = False
            return False
            
    def test_voice_activity_detection(self) -> bool:
        """Test Voice Activity Detection."""
        console.print("[blue]Testing Voice Activity Detection...[/blue]")
        
        try:
            import webrtcvad
            
            vad = webrtcvad.Vad(2)  # Aggressiveness level 2
            
            # Generate test audio (silence and noise)
            sample_rate = 16000
            frame_duration = 0.02  # 20ms
            frame_size = int(sample_rate * frame_duration)
            
            # Test with silence
            silence = np.zeros(frame_size, dtype=np.int16).tobytes()
            is_speech_silence = vad.is_speech(silence, sample_rate)
            
            # Test with noise
            noise = (np.random.randn(frame_size) * 1000).astype(np.int16).tobytes()
            is_speech_noise = vad.is_speech(noise, sample_rate)
            
            console.print(f"  Silence detected as speech: {is_speech_silence}")
            console.print(f"  Noise detected as speech: {is_speech_noise}")
            
            if not is_speech_silence:
                console.print("[green]✓ Voice Activity Detection working correctly[/green]")
                self.test_results['vad'] = True
                return True
            else:
                console.print("[red]✗ VAD incorrectly detecting silence as speech[/red]")
                self.test_results['vad'] = False
                return False
                
        except Exception as e:
            console.print(f"[red]✗ VAD test failed: {e}[/red]")
            self.test_results['vad'] = False
            return False
            
    def test_audio_handler(self) -> bool:
        """Test the complete audio handler system."""
        console.print("[blue]Testing MacAudioHandler...[/blue]")
        
        try:
            # Initialize audio handler
            config = AudioConfig(
                sample_rate=16000,
                chunk_duration=0.5,
                wake_word="robot"
            )
            
            self.audio_handler = MacAudioHandler(config)
            
            # Test audio input
            if not self.audio_handler.test_audio_input(duration=2.0):
                console.print("[red]✗ Audio handler input test failed[/red]")
                self.test_results['audio_handler'] = False
                return False
                
            console.print("[green]✓ MacAudioHandler test passed[/green]")
            self.test_results['audio_handler'] = True
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Audio handler test failed: {e}[/red]")
            self.test_results['audio_handler'] = False
            return False
            
    async def test_live_recognition(self, duration: float = 10.0) -> bool:
        """Test live voice recognition."""
        console.print(f"[blue]Testing live voice recognition for {duration} seconds...[/blue]")
        console.print("[yellow]Speak some commands like 'robot move left' or 'robot open gripper'[/yellow]")
        
        if not self.audio_handler:
            console.print("[red]✗ Audio handler not initialized[/red]")
            return False
            
        recognized_commands = []
        
        def command_callback(text: str):
            recognized_commands.append(text)
            console.print(f"[green]Recognized: '{text}'[/green]")
            
        try:
            # Start listening
            self.audio_handler.start_listening(command_callback)
            
            # Wait for the duration
            await asyncio.sleep(duration)
            
            # Stop listening
            self.audio_handler.stop_listening()
            
            console.print(f"[blue]Recognized {len(recognized_commands)} commands:[/blue]")
            for i, cmd in enumerate(recognized_commands, 1):
                console.print(f"  {i}. {cmd}")
                
            if recognized_commands:
                console.print("[green]✓ Live recognition test passed[/green]")
                self.test_results['live_recognition'] = True
                return True
            else:
                console.print("[yellow]⚠ No commands recognized (try speaking louder or closer to microphone)[/yellow]")
                self.test_results['live_recognition'] = False
                return False
                
        except Exception as e:
            console.print(f"[red]✗ Live recognition test failed: {e}[/red]")
            self.test_results['live_recognition'] = False
            return False
            
    def test_performance_benchmarks(self) -> bool:
        """Run performance benchmarks."""
        console.print("[blue]Running performance benchmarks...[/blue]")
        
        if not self.audio_handler:
            console.print("[red]✗ Audio handler not initialized[/red]")
            return False
            
        try:
            # Benchmark audio processing
            sample_rates = [16000, 22050, 44100]
            chunk_durations = [0.25, 0.5, 1.0]
            
            table = Table(title="Audio Processing Benchmarks")
            table.add_column("Sample Rate", style="cyan")
            table.add_column("Chunk Duration", style="yellow")
            table.add_column("Processing Time", style="green")
            table.add_column("Real-time Factor", style="magenta")
            
            for sr in sample_rates:
                for duration in chunk_durations:
                    # Generate test audio
                    audio_length = int(sr * duration)
                    test_audio = np.random.randn(audio_length).astype(np.float32)
                    
                    # Time transcription
                    start_time = time.time()
                    text = self.audio_handler.whisper.transcribe(test_audio)
                    processing_time = time.time() - start_time
                    
                    real_time_factor = duration / processing_time
                    
                    table.add_row(
                        f"{sr} Hz",
                        f"{duration}s",
                        f"{processing_time:.3f}s",
                        f"{real_time_factor:.2f}x"
                    )
                    
            console.print(table)
            
            console.print("[green]✓ Performance benchmarks completed[/green]")
            self.test_results['benchmarks'] = True
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Benchmark test failed: {e}[/red]")
            self.test_results['benchmarks'] = False
            return False
            
    def display_summary(self):
        """Display test summary."""
        table = Table(title="Voice System Test Summary")
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Notes", style="dim")
        
        status_map = {
            'audio_devices': "Audio Devices",
            'microphone_input': "Microphone Input", 
            'whisper_models': "Whisper Models",
            'vad': "Voice Activity Detection",
            'audio_handler': "Audio Handler",
            'live_recognition': "Live Recognition",
            'benchmarks': "Performance Benchmarks"
        }
        
        for key, name in status_map.items():
            if key in self.test_results:
                status = "✓ PASS" if self.test_results[key] else "✗ FAIL"
                color = "green" if self.test_results[key] else "red"
                table.add_row(name, f"[{color}]{status}[/{color}]", "")
            else:
                table.add_row(name, "[yellow]⏭ SKIPPED[/yellow]", "")
                
        console.print(table)
        
        # Overall result
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        
        if passed_tests == total_tests:
            console.print("[green bold]🎉 All tests passed! Voice system is ready.[/green bold]")
        else:
            console.print(f"[yellow]⚠ {passed_tests}/{total_tests} tests passed. Check failed tests above.[/yellow]")


async def main():
    parser = argparse.ArgumentParser(description="Test Saorse voice recognition system")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--live", action="store_true", help="Run live recognition test")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration for live test (seconds)")
    
    args = parser.parse_args()
    
    # Show banner
    console.print(Panel.fit(
        "[bold blue]Saorse Voice System Tester[/bold blue]\n"
        "[dim]Comprehensive testing of speech recognition components[/dim]",
        title="🎤 Voice Testing",
        border_style="blue"
    ))
    
    tester = VoiceSystemTester()
    
    # Run tests
    console.print("[blue]Starting voice system tests...[/blue]\n")
    
    # Core tests
    tester.test_audio_devices()
    tester.test_microphone_input()
    tester.test_whisper_models()
    tester.test_voice_activity_detection()
    tester.test_audio_handler()
    
    if not args.quick:
        if args.live:
            await tester.test_live_recognition(args.duration)
            
        if args.benchmark:
            tester.test_performance_benchmarks()
    
    # Show summary
    console.print("\n")
    tester.display_summary()
    
    # Recommendations
    console.print("\n[blue]Recommendations:[/blue]")
    if not tester.test_results.get('microphone_input', True):
        console.print("• Check microphone connection and permissions")
        console.print("• Try speaking louder or closer to the microphone")
        
    if not tester.test_results.get('whisper_models', True):
        console.print("• Run: python scripts/download_models.py")
        console.print("• Check internet connection for model downloads")
        
    console.print("• For best results, use the system in a quiet environment")
    console.print("• Speak clearly and wait for recognition feedback")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    asyncio.run(main())