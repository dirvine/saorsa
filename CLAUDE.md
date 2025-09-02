# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Saorsa is a sophisticated voice-controlled system for SO-101 robot arms, featuring natural language processing, AI-powered command interpretation, and computer vision integration, optimized for macOS with Apple Silicon.

## Build and Development Commands

### Installation and Setup
```bash
# Initial setup (installs all dependencies)
./scripts/install_mac.sh

# Download AI models
python scripts/download_models.py --all  # Download all models
python scripts/download_models.py --basic  # Download basic models only
```

### Running the Application
```bash
# Basic launch with leader robot only
./launch.sh /dev/tty.usbserial-FT1234

# Dual robot setup
./launch.sh /dev/tty.usbserial-FT1234 /dev/tty.usbserial-FT5678

# Direct Python execution with different modes
python src/main_mac.py run -l /dev/tty.usbserial-FT1234 --mode basic       # Voice only
python src/main_mac.py run -l /dev/tty.usbserial-FT1234 --mode ai          # AI-enhanced
python src/main_mac.py run -l /dev/tty.usbserial-FT1234 --mode multimodal  # Voice + vision
```

### Testing and Debugging
```bash
# System components testing
python src/main_mac.py test-audio        # Test microphone and speech recognition
python src/main_mac.py test-robot -p /dev/tty.usbserial-FT1234  # Test robot connection
python src/main_mac.py test-camera       # Test camera access
python src/main_mac.py status            # Show system status

# Demo features
python src/main_mac.py demo-ai           # Demo AI command processing
python src/main_mac.py demo-vision       # Demo computer vision capabilities

# Unit tests
python -m pytest tests/
```

### Development Tools
```bash
# Code formatting and linting
black src/                # Format Python code
flake8 src/              # Lint Python code
mypy src/                # Type checking

# Virtual environment management
source venv/bin/activate  # Activate virtual environment
deactivate              # Deactivate virtual environment
```

## Architecture Overview

### Core Components

1. **Main Application Entry (`src/main_mac.py`)**
   - CLI interface using Click framework
   - Command routing and orchestration
   - Three operation modes: basic, AI-enhanced, multimodal

2. **Audio Processing (`src/mac_audio_handler.py`)**
   - OpenAI Whisper for speech recognition
   - Voice Activity Detection (VAD)
   - Wake word detection ("robot")
   - macOS-specific audio APIs (pyobjc)

3. **Robot Control (`src/robot_controller_m3.py`)**
   - Dynamixel SDK for motor control
   - Safety monitoring and workspace limits
   - Emergency stop functionality
   - Support for leader/follower dual-arm setup

4. **AI Command Processing (`src/ai_command_processor.py`)**
   - Local Hugging Face models (SmolLM2, Qwen2.5, Phi-3)
   - Natural language understanding
   - Context-aware command interpretation
   - Metal Performance Shaders (MPS) optimization for Apple Silicon

5. **Computer Vision (`src/object_detector.py`)**
   - DETR and YOLO models for object detection
   - Real-time processing pipeline
   - Spatial reasoning capabilities

6. **Multimodal Interface (`src/multimodal_interface.py`)**
   - Combines voice commands with visual context
   - Spatial reference resolution ("pick up the object on the left")
   - Object tracking across frames

### Configuration System

Configuration files in `configs/`:
- `default.yaml` - Base configuration for all systems
- `mac_m3.yaml` - Apple Silicon specific optimizations
- `voice_commands.yaml` - Voice command mappings

Key configuration sections:
- Audio settings (sample rate, VAD, wake word)
- Robot parameters (workspace limits, motor configs)
- Safety thresholds (temperature, current, velocity)
- AI model selection and optimization
- Vision processing parameters

### Model Management

AI models are stored in `models/` directory and downloaded on-demand. The system supports:
- Language models: SmolLM2 (135M, 360M, 1.7B), Qwen2.5, Phi-3
- Vision models: DETR (ResNet-50/101), YOLO (v8n/s/m/l)

### Safety Architecture

Multi-layered safety system:
1. Hardware limits enforced by Dynamixel motors
2. Software workspace boundaries
3. Emergency stop on voice commands
4. Temperature and current monitoring
5. Collision detection via force feedback

## Key Design Patterns

1. **Async/Await Pattern**: Used throughout for non-blocking I/O operations
2. **Command Pattern**: Commands encapsulated as `CommandIntent` objects
3. **Observer Pattern**: Safety monitor observes robot state changes
4. **Pipeline Pattern**: Audio → Speech Recognition → NLP → Command → Robot
5. **Strategy Pattern**: Different model managers for various AI models

## Development Guidelines

1. **Performance Optimization**:
   - Use MPS (Metal Performance Shaders) for AI inference on Apple Silicon
   - Implement model quantization (fp16) for faster inference
   - Cache frequently used models in memory

2. **Error Handling**:
   - All robot commands wrapped in try-except blocks
   - Emergency stop always accessible
   - Graceful degradation when AI models unavailable

3. **Testing**:
   - Component isolation tests for each module
   - Integration tests for voice → robot pipeline
   - Performance benchmarks for AI models

4. **Adding New Features**:
   - Voice commands: Edit `CommandProcessor.basic_commands` in `main_mac.py`
   - AI behaviors: Extend `AICommandProcessor` in `ai_command_processor.py`
   - Vision capabilities: Add to `ObjectDetector` in `object_detector.py`

## Dependencies

Critical Python packages:
- `torch` - PyTorch for AI models (with MPS support)
- `transformers` - Hugging Face model library
- `openai-whisper` - Speech recognition
- `dynamixel-sdk` - Robot motor control
- `sounddevice` - Audio capture
- `opencv-python` - Computer vision
- `pyobjc-framework-*` - macOS-specific APIs

## Common Tasks

### Adding a new voice command
1. Add mapping in `CommandProcessor.basic_commands` or `movement_commands`
2. Implement handler method in `RobotController`
3. Add safety checks in `SafetyMonitor`
4. Update `voice_commands.yaml` configuration

### Switching AI models
1. Edit `ai.default_model` in configuration
2. Ensure model is in `ai.available_models` list
3. Run `python scripts/download_models.py --model <model-name>`

### Calibrating robot
```bash
python scripts/calibrate_robot.py /dev/tty.usbserial-FT1234
```

### Monitoring performance
Performance metrics are logged to `logs/metrics.json` every 5 minutes. Real-time monitoring available through `PerformanceMonitor` class.