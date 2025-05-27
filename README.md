# Saorse: Advanced Voice-Controlled SO-101 Robot Arms

A sophisticated natural language interface for controlling Hugging Face's SO-101 robot arms, featuring voice recognition, AI-powered command interpretation, and computer vision integration. Optimized for macOS with Apple Silicon.

## Overview

Saorse provides three levels of robot control sophistication:

- **Phase 1**: Core voice control with basic movement commands
- **Phase 2**: AI-enhanced natural language processing with context awareness  
- **Phase 3**: Multimodal interface combining voice commands with computer vision

## Features

### Core Capabilities
- 🎤 **Natural Language Control**: Speak commands in plain English
- 🤖 **Multi-Robot Support**: Control single or dual SO-101 robot arms
- 🛡️ **Safety First**: Built-in workspace limits and emergency controls
- ⚡ **Mac M3 Optimized**: Leverages Apple Silicon for efficient processing
- 🔧 **Modular Architecture**: Extensible for custom commands and behaviors

### AI & Vision Features
- 🧠 **Local AI Models**: Advanced command interpretation using Hugging Face transformers
- 👁️ **Computer Vision**: Real-time object detection and spatial reasoning
- 🎯 **Spatial References**: "Pick up the red block on the left"
- 📝 **Context Awareness**: Remember objects and locations across commands
- 🖼️ **Visual Feedback**: Real-time overlays showing detected objects and robot status

## Quick Start

### 1. System Requirements

#### Hardware
- **Computer**: macOS with Apple Silicon (M1, M2, M3, or later)
- **Memory**: 16GB RAM minimum (32GB recommended for advanced AI features)
- **Storage**: 25GB free space for models and data
- **Robot**: SO-101 Robot Arms (1-2 units)
- **Audio**: Built-in microphone or external USB microphone
- **Camera**: Built-in camera, USB camera, or iPhone (via Continuity Camera)

#### Software
- **OS**: macOS 13.0+ (Ventura or later)
- **Python**: 3.11 or higher
- **Homebrew**: For system dependencies

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/dirvine/saorse.git
cd saorse

# Run the automated installation script
./scripts/install_mac.sh

# Download AI models (optional for basic usage)
python scripts/download_models.py --basic

# For full AI features, download all models
python scripts/download_models.py --all
```

### 3. Robot Setup

```bash
# Find your robot's serial port
ls /dev/tty.usbserial-*

# Test robot connection (replace with your actual port)
python src/main_mac.py test-robot --port /dev/tty.usbserial-FT1234

# Calibrate the robot if needed
python scripts/calibrate_robot.py /dev/tty.usbserial-FT1234
```

### 4. Test Systems

```bash
# Test audio input and speech recognition
python src/main_mac.py test-audio

# Test camera system (for Phase 3 features)
python src/main_mac.py test-camera

# Check overall system status
python src/main_mac.py status
```

### 5. Launch Voice Control

Choose your operation mode:

```bash
# Basic voice control (Phase 1)
python src/main_mac.py run -l /dev/tty.usbserial-FT1234

# AI-enhanced processing (Phase 2)
python src/main_mac.py run -l /dev/tty.usbserial-FT1234 --mode ai

# Full multimodal with vision (Phase 3)
python src/main_mac.py run -l /dev/tty.usbserial-FT1234 --mode multimodal

# Dual robot setup
python src/main_mac.py run -l /dev/tty.usbserial-FT1234 -f /dev/tty.usbserial-FT5678 --mode multimodal
```

## Voice Commands

### Basic Commands (Phase 1)

#### Movement
- "move left" / "move right"
- "move forward" / "move back" / "move backward"
- "move up" / "move down"
- "turn left" / "turn right"

#### Gripper Control
- "open gripper" / "open"
- "close gripper" / "close" / "grab"
- "release"

#### Position Commands
- "home" / "home position"
- "ready" / "ready position"

#### Control
- "stop"
- "halt" / "emergency" / "emergency stop" / "freeze"

### AI-Enhanced Commands (Phase 2)

#### Object Manipulation
- "pick up the red block"
- "put it over there"
- "move it to the left side"
- "place it on the table"

#### Task-Level Commands
- "stack the blocks"
- "organize the objects by color"
- "arrange the items neatly"
- "sort everything by size"

#### Context-Aware References
- "move that to the center" (refers to previously mentioned object)
- "put this next to the blue cup" (spatial relationships)
- "stack it on top of that" (object references)

### Multimodal Commands (Phase 3)

#### Spatial References with Vision
- "pick up the object on the left"
- "grab the largest item"
- "move the cup next to the bottle"
- "stack the blocks by size"

#### Visual Confirmation
- "show me what you can see"
- "identify the objects on the table"
- "point to the red object"

## AI Model Configuration

### Model Selection

The system supports multiple AI models for different use cases:

#### Language Models (Phase 2)

**Lightweight Models (8GB+ RAM)**:
```yaml
# In configs/default.yaml
ai_models:
  primary: "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # Fast, efficient
  fallback: "microsoft/DialoGPT-medium"           # Conversation
```

**Balanced Models (16GB+ RAM)**:
```yaml
ai_models:
  primary: "Qwen/Qwen2.5-3B-Instruct"            # Good balance
  fallback: "microsoft/Phi-3.5-mini-instruct"     # Reliable
```

**High-Performance Models (32GB+ RAM)**:
```yaml
ai_models:
  primary: "Qwen/Qwen2.5-14B-Instruct"           # Best quality
  fallback: "meta-llama/Llama-3.2-3B-Instruct"   # Backup
```

#### Vision Models (Phase 3)

**Object Detection Models**:
```yaml
vision_models:
  object_detection: "facebook/detr-resnet-50"     # Default, reliable
  # alternatives:
  # object_detection: "facebook/detr-resnet-101"  # Higher accuracy
  # object_detection: "microsoft/table-transformer-object-detection"  # Specialized
```

**YOLO Models** (if ultralytics installed):
```yaml
vision_models:
  object_detection: "yolov8n"  # Fastest
  # object_detection: "yolov8s"  # Balanced
  # object_detection: "yolov8m"  # More accurate
  # object_detection: "yolov8l"  # Best accuracy
```

### Custom Model Configuration

Edit `configs/mac_m3.yaml` for your specific setup:

```yaml
# Model optimization settings
model_optimization:
  enable_mps: true              # Use Metal Performance Shaders
  enable_torch_compile: true    # PyTorch 2.0 optimizations
  fp16_inference: true          # Half precision for speed
  batch_size: 1                 # Adjust based on available memory

# AI processing settings  
ai_processing:
  confidence_threshold: 0.7     # Command confidence threshold
  context_window: 10           # Number of previous commands to remember
  enable_context_tracking: true # Track objects and locations
  
# Vision processing settings
vision_processing:
  detection_confidence: 0.6     # Object detection threshold
  detection_fps: 10            # Frames per second for detection
  max_detections: 100          # Maximum objects per frame
  enable_tracking: true        # Track objects across frames
```

### Downloading Specific Models

```bash
# Download specific model sets
python scripts/download_models.py --language-only      # Just language models
python scripts/download_models.py --vision-only        # Just vision models
python scripts/download_models.py --lightweight        # Lightweight models only

# Download specific models
python scripts/download_models.py --model "Qwen/Qwen2.5-3B-Instruct"
python scripts/download_models.py --model "facebook/detr-resnet-101"

# Check what's downloaded
python scripts/download_models.py --check
python scripts/download_models.py --list-available
```

## Demo and Testing

### AI Capabilities Demo

```bash
# Test AI command processing
python src/main_mac.py demo-ai

# Test with specific models
python src/main_mac.py demo-ai --model "Qwen/Qwen2.5-3B-Instruct"
```

### Vision System Demo

```bash
# Test computer vision capabilities
python src/main_mac.py demo-vision

# Test camera access and permissions
python src/main_mac.py test-camera
```

### Integration Testing

```bash
# Test multimodal integration (voice + vision)
python src/main_mac.py run --mode multimodal --demo

# Run system diagnostics
python src/main_mac.py status --verbose
```

## Advanced Configuration

### Audio Settings

```yaml
# In configs/default.yaml
audio:
  sample_rate: 16000
  chunk_size: 1024
  whisper_model: "base"           # tiny, base, small, medium, large
  vad_threshold: 0.5              # Voice activity detection
  silence_timeout: 2.0            # Seconds of silence to end recording
  phrase_timeout: 3.0             # Maximum phrase length
```

### Robot Control Settings

```yaml
robot_control:
  movement_speed: 50              # Joint movement speed (0-100)
  acceleration: 30                # Movement acceleration
  gripper_force: 50               # Gripper closing force
  position_tolerance: 5           # Position accuracy in degrees
  
safety:
  workspace_bounds:               # Workspace limits in degrees
    joint1: [-150, 150]
    joint2: [-90, 90]
    joint3: [-90, 90]
    joint4: [-180, 180]
    joint5: [-90, 90]
    joint6: [-180, 180]
  emergency_stop_on_error: true
  max_temperature: 70             # Maximum motor temperature (°C)
```

### Vision System Settings

```yaml
vision:
  camera:
    resolution: [1280, 720]       # Camera resolution
    fps: 30                       # Camera framerate
    enable_continuity_camera: true # iPhone camera support
    
  detection:
    confidence_threshold: 0.6     # Detection confidence
    nms_threshold: 0.5           # Non-maximum suppression
    max_detections: 100          # Max objects per frame
    
  display:
    show_detections: true        # Show detection overlays
    show_robot_status: true      # Show robot information
    show_workspace_bounds: true  # Show workspace limits
    overlay_alpha: 0.7          # Overlay transparency
```

## Troubleshooting

### Installation Issues

#### PyTorch Installation
```bash
# Ensure correct PyTorch for Apple Silicon
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio

# Verify MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

#### Model Download Issues
```bash
# Clear model cache and retry
rm -rf ~/.cache/huggingface/
python scripts/download_models.py --basic

# Download with specific cache directory
export HF_HOME=/path/to/large/storage
python scripts/download_models.py --all
```

### Runtime Issues

#### Audio Problems
```bash
# Check microphone permissions
# System Preferences > Security & Privacy > Microphone

# Test audio devices
python -c "
import sounddevice as sd
print('Audio devices:', sd.query_devices())
"

# Test Whisper directly
python -c "
import whisper
model = whisper.load_model('base')
print('Whisper model loaded successfully')
"
```

#### Robot Connection Issues
```bash
# Check USB serial permissions
ls -la /dev/tty.usbserial-*

# Test with different baud rates
python scripts/test_robot_connection.py --port /dev/tty.usbserial-FT1234 --baud 1000000

# Check motor health
python scripts/diagnose_motors.py /dev/tty.usbserial-FT1234
```

#### Vision System Issues
```bash
# Check camera permissions
# System Preferences > Security & Privacy > Camera

# Test camera access
python -c "
from src.mac_camera_handler import MacCameraHandler
handler = MacCameraHandler()
print('Camera access:', handler.test_camera_access())
"

# Test object detection
python src/object_detector.py  # Runs built-in demo
```

#### Performance Issues
```bash
# Monitor system resources
python scripts/monitor_performance.py

# Check model memory usage
python -c "
import torch
print(f'GPU memory: {torch.mps.current_allocated_memory()/1024**3:.2f} GB')
"

# Optimize for lower memory usage
# Edit configs/mac_m3.yaml:
# model_optimization:
#   fp16_inference: true
#   low_memory_mode: true
```

### Common Error Messages

#### "MPS not available"
- Update to macOS 12.3+ and PyTorch 1.12+
- Ensure you're using Apple Silicon Mac

#### "Model not found"
- Run `python scripts/download_models.py --check`
- Download missing models with appropriate command

#### "Camera permission denied"
- Enable camera access in System Preferences > Security & Privacy > Camera

#### "Robot connection timeout"
- Check USB cable and robot power
- Verify correct serial port with `ls /dev/tty.usbserial-*`
- Try different baud rate in robot configuration

## Development

### Project Structure

```
saorse/
├── src/                          # Main source code
│   ├── main_mac.py              # Application entry point
│   ├── mac_audio_handler.py     # Voice recognition
│   ├── robot_controller_m3.py   # Robot control
│   ├── ai_command_processor.py  # AI command interpretation
│   ├── mac_camera_handler.py    # Camera integration
│   ├── object_detector.py       # Computer vision
│   ├── visual_feedback.py       # Visual overlays
│   ├── multimodal_interface.py  # Voice + vision integration
│   ├── context_manager.py       # Context awareness
│   ├── model_manager.py         # AI model management
│   ├── mps_optimizer.py         # Apple Silicon optimization
│   └── utils/                   # Utility modules
├── configs/                     # Configuration files
├── scripts/                     # Setup and utility scripts
├── models/                      # Downloaded AI models
├── logs/                        # Application logs
└── docs/                        # Documentation
```

### Running Tests

```bash
# Unit tests
python -m pytest tests/

# Component tests
python scripts/test_audio.py
python scripts/test_robot.py
python scripts/test_vision.py

# Integration tests
python scripts/test_multimodal.py

# Performance benchmarks
python scripts/benchmark_models.py
```

### Adding Custom Commands

1. **Basic Commands**: Edit `src/main_mac.py` in `CommandProcessor.basic_commands`

2. **AI Commands**: Train/fine-tune models or add examples in `configs/ai_examples.yaml`

3. **Multimodal Commands**: Extend `src/multimodal_interface.py` spatial reference resolution

### Configuration Management

All configurations are in YAML format under `configs/`:

- `default.yaml` - Base configuration for all systems
- `mac_m3.yaml` - Apple Silicon optimizations  
- `robot_configs.yaml` - Robot-specific settings
- `voice_commands.yaml` - Voice command mappings
- `ai_models.yaml` - AI model configurations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Code Style

- Follow PEP 8 formatting
- Use type hints for all functions
- Add docstrings for public methods
- Include unit tests for new features

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Changelog

### Version 1.3.0 (Current)
- ✅ Phase 3: Computer vision and multimodal interface
- ✅ Real-time object detection with local models
- ✅ Spatial reference resolution ("pick up the object on the left")
- ✅ Visual feedback with detection overlays
- ✅ iPhone Continuity Camera support
- ✅ Enhanced context management with visual data

### Version 1.2.0
- ✅ Phase 2: AI integration with local Hugging Face models
- ✅ Context-aware command processing
- ✅ Advanced natural language understanding
- ✅ Mac M3 MPS optimization for AI models

### Version 1.1.0
- ✅ Phase 1: Core voice control and robot integration
- ✅ OpenAI Whisper speech recognition
- ✅ SO-101 robot arm control
- ✅ Safety monitoring and emergency stops

## Documentation

- [Hardware Setup Guide](docs/hardware_setup.md)
- [Software Installation Guide](docs/software_setup.md)
- [Voice Commands Reference](docs/voice_commands.md)
- [AI Model Guide](docs/ai_models.md)
- [Vision System Guide](docs/vision_setup.md)
- [API Documentation](docs/api_reference.md)
- [Troubleshooting Guide](docs/troubleshooting.md)