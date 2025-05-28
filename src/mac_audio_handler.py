#!/usr/bin/env python3
"""
macOS Audio Handler for Saorsa Voice Recognition System

This module provides voice input processing using macOS Core Audio frameworks
and OpenAI Whisper for speech-to-text conversion, optimized for Apple Silicon M3.
"""

import asyncio
import logging
import threading
import time
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
from queue import Queue, Empty

import torch
import whisper
import sounddevice as sd
import numpy as np
import webrtcvad
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 16000
    chunk_duration: float = 0.5
    channels: int = 1
    dtype: str = 'float32'
    wake_word: str = "robot"
    vad_aggressiveness: int = 2
    silence_timeout: float = 2.0
    min_speech_duration: float = 0.5


class VoiceActivityDetector:
    """Voice Activity Detection using WebRTC VAD."""
    
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        
    def is_speech(self, audio_data: np.ndarray) -> bool:
        """Check if audio contains speech."""
        # Convert float32 to int16 for WebRTC VAD
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # WebRTC VAD expects specific frame sizes
        frame_size = int(self.sample_rate * 0.02)  # 20ms frames
        
        if len(audio_int16) < frame_size:
            return False
            
        # Take the first complete frame
        frame = audio_bytes[:frame_size * 2]  # 2 bytes per int16
        
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception:
            return False


class WhisperProcessor:
    """Whisper model processor optimized for Mac M3."""
    
    def __init__(self, model_size: str = "base", device: str = "mps"):
        self.device = device
        self.model_size = model_size
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load Whisper model with MPS acceleration."""
        try:
            console.print(f"[yellow]Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(
                self.model_size,
                device=self.device if torch.backends.mps.is_available() else "cpu"
            )
            console.print(f"[green]Whisper model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            # Fallback to CPU
            self.device = "cpu"
            self.model = whisper.load_model(self.model_size, device="cpu")
            console.print("[yellow]Fallback: Using CPU for Whisper")
            
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio to text."""
        if self.model is None:
            return ""
            
        try:
            # Whisper expects audio at 16kHz
            result = self.model.transcribe(
                audio_data,
                language="en",
                task="transcribe",
                fp16=False  # Disable fp16 for MPS compatibility
            )
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""


class MacAudioHandler:
    """
    Main audio handler for macOS voice recognition.
    
    Provides continuous audio capture, voice activity detection,
    and speech-to-text conversion using Whisper.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.whisper = WhisperProcessor()
        self.vad = VoiceActivityDetector(self.config.vad_aggressiveness)
        
        self.is_listening = False
        self.audio_queue = Queue()
        self.callback_function: Optional[Callable[[str], None]] = None
        
        self._audio_thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None
        
        # Audio buffer for accumulating speech
        self.audio_buffer = []
        self.last_speech_time = 0
        self.recording_speech = False
        
    def start_listening(self, callback: Callable[[str], None]) -> None:
        """Start continuous voice recognition."""
        if self.is_listening:
            logger.warning("Already listening")
            return
            
        self.callback_function = callback
        self.is_listening = True
        
        # Start audio capture thread
        self._audio_thread = threading.Thread(
            target=self._audio_capture_loop,
            daemon=True
        )
        self._audio_thread.start()
        
        # Start audio processing thread
        self._processing_thread = threading.Thread(
            target=self._audio_processing_loop,
            daemon=True
        )
        self._processing_thread.start()
        
        console.print(f"[green]🎤 Listening for voice commands (wake word: '{self.config.wake_word}')")
        
    def stop_listening(self) -> None:
        """Stop voice recognition."""
        if not self.is_listening:
            return
            
        self.is_listening = False
        
        # Wait for threads to finish
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.0)
            
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
            
        console.print("[yellow]🔇 Stopped listening")
        
    def set_wake_word(self, word: str) -> None:
        """Set the wake word for activation."""
        self.config.wake_word = word.lower()
        console.print(f"[blue]Wake word set to: '{word}'")
        
    def set_language(self, lang: str) -> None:
        """Set the language for recognition."""
        # Note: This would require reloading the Whisper model
        # For now, we'll just log the change
        console.print(f"[blue]Language preference set to: '{lang}'")
        
    def _audio_capture_loop(self) -> None:
        """Continuous audio capture loop."""
        chunk_size = int(self.config.sample_rate * self.config.chunk_duration)
        
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio input status: {status}")
            if self.is_listening:
                # Convert to mono if stereo
                audio_data = indata[:, 0] if len(indata.shape) > 1 else indata
                self.audio_queue.put(audio_data.copy())
                
        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=chunk_size,
                callback=audio_callback
            ):
                while self.is_listening:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            console.print(f"[red]Audio capture failed: {e}")
            
    def _audio_processing_loop(self) -> None:
        """Process captured audio for speech detection and transcription."""
        while self.is_listening:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Check for voice activity
                has_speech = self.vad.is_speech(audio_chunk)
                current_time = time.time()
                
                if has_speech:
                    if not self.recording_speech:
                        # Start recording speech
                        self.recording_speech = True
                        self.audio_buffer = []
                        console.print("[cyan]🗣️  Speech detected...")
                        
                    self.audio_buffer.append(audio_chunk)
                    self.last_speech_time = current_time
                    
                elif self.recording_speech:
                    # Check if silence timeout reached
                    silence_duration = current_time - self.last_speech_time
                    
                    if silence_duration >= self.config.silence_timeout:
                        # Process accumulated speech
                        self._process_speech_buffer()
                        self.recording_speech = False
                        self.audio_buffer = []
                    else:
                        # Continue recording during short pauses
                        self.audio_buffer.append(audio_chunk)
                        
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                
    def _process_speech_buffer(self) -> None:
        """Process accumulated speech audio."""
        if not self.audio_buffer:
            return
            
        # Concatenate audio chunks
        audio_data = np.concatenate(self.audio_buffer)
        
        # Check minimum duration
        duration = len(audio_data) / self.config.sample_rate
        if duration < self.config.min_speech_duration:
            return
            
        console.print("[yellow]🔄 Processing speech...")
        
        # Transcribe with Whisper
        text = self.whisper.transcribe(audio_data)
        
        if text:
            console.print(f"[green]📝 Recognized: '{text}'")
            
            # Check for wake word or direct command
            if self._should_process_command(text):
                if self.callback_function:
                    try:
                        self.callback_function(text)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
        else:
            console.print("[dim]🔇 No speech recognized")
            
    def _should_process_command(self, text: str) -> bool:
        """Determine if the recognized text should trigger a command."""
        text_lower = text.lower()
        
        # Always process if wake word is present
        if self.config.wake_word in text_lower:
            return True
            
        # Check for emergency words
        emergency_words = ["stop", "halt", "emergency"]
        for word in emergency_words:
            if word in text_lower:
                return True
                
        return False
        
    def get_audio_devices(self) -> Dict[str, Any]:
        """Get available audio input devices."""
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
                
        return {
            'default_device': sd.default.device[0],
            'input_devices': input_devices
        }
        
    def test_audio_input(self, duration: float = 3.0) -> bool:
        """Test audio input functionality."""
        try:
            console.print(f"[yellow]Testing audio input for {duration} seconds...")
            
            audio_data = sd.rec(
                int(duration * self.config.sample_rate),
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype
            )
            sd.wait()
            
            # Check if we got audio data
            if np.max(np.abs(audio_data)) > 0.01:
                console.print("[green]✓ Audio input test successful")
                
                # Test transcription
                text = self.whisper.transcribe(audio_data.flatten())
                if text:
                    console.print(f"[green]✓ Transcription test: '{text}'")
                else:
                    console.print("[yellow]⚠️  No speech detected in test")
                    
                return True
            else:
                console.print("[red]✗ No audio input detected")
                return False
                
        except Exception as e:
            console.print(f"[red]✗ Audio test failed: {e}")
            return False


async def main():
    """Example usage of the MacAudioHandler."""
    
    def command_callback(text: str):
        console.print(f"[bold green]COMMAND: {text}")
        
    handler = MacAudioHandler()
    
    # Test audio
    if not handler.test_audio_input():
        console.print("[red]Audio test failed. Check your microphone.")
        return
        
    # Start listening
    handler.start_listening(command_callback)
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        console.print("[yellow]Shutting down...")
    finally:
        handler.stop_listening()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())