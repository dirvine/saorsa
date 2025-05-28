# Software Setup Guide

This comprehensive guide covers installing and configuring the Saorsa voice-controlled robot system on macOS with Apple Silicon.

## Prerequisites

### System Requirements

#### Operating System
- **macOS 13.0** (Ventura) or later
- **Apple Silicon**: M1, M2, M3, or later processor
- **Xcode Command Line Tools**: Required for compilation

#### Developer Tools
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Python Environment

#### Python Installation
```bash
# Install Python 3.11+ via Homebrew
brew install python@3.11

# Verify installation
python3.11 --version  # Should show 3.11.x or later
```

#### Virtual Environment Setup
```bash
# Create virtual environment
python3.11 -m venv saorsa-env

# Activate environment
source saorsa-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

## Installation Methods

### Method 1: Automated Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/dirvine/saorsa.git
cd saorsa

# Run automated installation script
./scripts/install_mac.sh

# Activate the created virtual environment
source venv/bin/activate
```

The automated script will:
- Set up Python virtual environment
- Install all required dependencies
- Configure macOS permissions
- Download basic AI models
- Verify installation

### Method 2: Manual Installation

#### Step 1: Clone Repository
```bash
git clone https://github.com/dirvine/saorsa.git
cd saorsa
```

#### Step 2: Create Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

#### Step 3: Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

#### Step 4: Install System Dependencies
```bash
# Install system libraries via Homebrew
brew install portaudio
brew install ffmpeg
brew install cmake
```

## Dependency Configuration

### PyTorch Installation

#### Verify Apple Silicon PyTorch
```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

#### Reinstall if Needed
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Install latest PyTorch for Apple Silicon
pip install torch torchvision torchaudio
```

### Hugging Face Configuration

#### Authentication (Optional)
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login for access to gated models (optional)
huggingface-cli login
```

#### Cache Configuration
```bash
# Set custom cache directory (optional)
export HF_HOME=/path/to/large/storage/huggingface
echo 'export HF_HOME=/path/to/large/storage/huggingface' >> ~/.zshrc
```

### macOS Permissions

#### Audio Permissions
1. **System Preferences** → **Security & Privacy** → **Privacy**
2. **Microphone** → Add Terminal, Python, and your terminal app
3. **Restart** terminal application

#### Camera Permissions (Phase 3)
1. **System Preferences** → **Security & Privacy** → **Privacy**  
2. **Camera** → Add Terminal, Python, and your terminal app
3. **Restart** terminal application

#### Verify Permissions
```bash
# Test audio permissions
python src/main_mac.py test-audio

# Test camera permissions (Phase 3)
python src/main_mac.py test-camera
```

## AI Model Setup

### Basic Model Download

#### Essential Models
```bash
# Download basic models for Phase 1 operation
python scripts/download_models.py --basic

# This downloads:
# - OpenAI Whisper base model
# - Basic language model for Phase 2
```

#### Full Model Suite
```bash
# Download all models for complete functionality
python scripts/download_models.py --all

# This downloads:
# - Multiple Whisper models (tiny, base, small, medium)
# - Language models (SmolLM2, Qwen2.5, Phi-3)
# - Vision models (DETR, YOLO variants)
```

### Model Selection by System Specifications

#### 16GB RAM Systems
```bash
# Lightweight models optimized for 16GB RAM
python scripts/download_models.py --lightweight

# Specific efficient models
python scripts/download_models.py --model "HuggingFaceTB/SmolLM2-1.7B-Instruct"
python scripts/download_models.py --model "microsoft/Phi-3.5-mini-instruct"
```

#### 32GB+ RAM Systems  
```bash
# High-performance models for systems with ample RAM
python scripts/download_models.py --high-performance

# Specific large models
python scripts/download_models.py --model "Qwen/Qwen2.5-14B-Instruct"
python scripts/download_models.py --model "meta-llama/Llama-3.2-3B-Instruct"
```

#### Vision Models (Phase 3)
```bash
# Download computer vision models
python scripts/download_models.py --vision-only

# Specific vision models
python scripts/download_models.py --model "facebook/detr-resnet-50"
python scripts/download_models.py --model "facebook/detr-resnet-101"
```

### Model Status and Management

#### Check Downloaded Models
```bash
# List downloaded models
python scripts/download_models.py --check

# List all available models
python scripts/download_models.py --list-available

# Check model sizes and locations
python scripts/download_models.py --info
```

#### Model Cache Management
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/

# Check cache size
du -sh ~/.cache/huggingface/

# Move cache to external storage
mv ~/.cache/huggingface/ /Volumes/ExternalDrive/huggingface-cache/
ln -s /Volumes/ExternalDrive/huggingface-cache/ ~/.cache/huggingface
```

## Configuration Setup

### Basic Configuration

#### Default Configuration
```bash
# Copy default configuration
cp configs/default.yaml configs/local.yaml

# Edit for your system
nano configs/local.yaml
```

#### Key Configuration Options
```yaml
# Audio settings
audio:
  whisper_model: "base"        # tiny, base, small, medium, large
  sample_rate: 16000
  vad_threshold: 0.5

# Robot settings  
robot:
  leader_port: "/dev/tty.usbserial-FT1234"
  follower_port: null          # Set for dual robot setup
  baud_rate: 1000000

# AI settings (Phase 2)
ai_models:
  primary: "HuggingFaceTB/SmolLM2-1.7B-Instruct"
  fallback: "microsoft/DialoGPT-medium"
  
# Vision settings (Phase 3)
vision:
  object_detection: "facebook/detr-resnet-50"
  camera_resolution: [1280, 720]
  detection_confidence: 0.6
```

### Apple Silicon Optimization

#### MPS Configuration
```yaml
# configs/mac_m3.yaml
model_optimization:
  enable_mps: true              # Use Metal Performance Shaders
  enable_torch_compile: true    # PyTorch 2.0 compilation
  fp16_inference: true          # Half precision for speed
  batch_size: 1                 # Optimize for single requests

performance:
  memory_fraction: 0.8          # Use 80% of available memory
  cpu_threads: 8                # Adjust based on CPU cores
  priority: "high"              # Process priority
```

#### Memory Optimization
```yaml
# For systems with limited RAM
memory_optimization:
  low_memory_mode: true
  model_offloading: true
  gradient_checkpointing: true
  attention_slicing: true
```

## System Validation

### Installation Verification

#### Basic System Test
```bash
# Check system status
python src/main_mac.py status

# Expected output:
# ✓ Audio Input: X devices
# ✓ PyTorch MPS: Available
# ✓ Hugging Face: X models downloaded
# ✓ Camera: X devices (if Phase 3)
```

#### Component Tests
```bash
# Test audio system
python src/main_mac.py test-audio

# Test robot connection (with actual robot)
python src/main_mac.py test-robot --port /dev/tty.usbserial-FT1234

# Test camera system (Phase 3)
python src/main_mac.py test-camera
```

### AI Model Validation

#### Test Language Models
```bash
# Test AI command processing
python src/main_mac.py demo-ai

# Test specific model
python scripts/test_model.py --model "HuggingFaceTB/SmolLM2-1.7B-Instruct"
```

#### Test Vision Models (Phase 3)
```bash
# Test computer vision
python src/main_mac.py demo-vision

# Test object detection
python src/object_detector.py
```

### Performance Benchmarking

#### System Performance
```bash
# Run performance benchmarks
python scripts/benchmark_models.py

# Monitor system resources
python scripts/monitor_performance.py
```

#### Expected Performance Metrics
- **Voice Recognition**: < 2 seconds latency
- **AI Processing**: < 5 seconds for complex commands
- **Object Detection**: > 10 FPS for real-time operation
- **Memory Usage**: < 80% of available RAM

## Development Setup

### Development Dependencies

#### Install Development Tools
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### Code Quality Tools
```bash
# Format code
black src/ scripts/

# Check linting
flake8 src/ scripts/

# Type checking
mypy src/
```

### Testing Framework

#### Run Tests
```bash
# Unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/

# Performance tests
python -m pytest tests/performance/ --benchmark-only
```

#### Test Coverage
```bash
# Generate coverage report
pytest --cov=src tests/
pytest --cov=src --cov-report=html tests/
```

### Development Workflow

#### Git Hooks
```bash
# Pre-commit configuration (.pre-commit-config.yaml)
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

#### Environment Variables
```bash
# Development environment variables
export SAORSE_ENV=development
export SAORSE_LOG_LEVEL=DEBUG
export SAORSE_CONFIG_PATH=configs/development.yaml
```

## Troubleshooting Installation

### Common Issues

#### Python Version Problems
```bash
# Check Python version
python --version

# Use specific Python version
python3.11 -m pip install -r requirements.txt
```

#### PyTorch Installation Issues
```bash
# Clean PyTorch installation
pip uninstall torch torchvision torchaudio
pip cache purge
pip install torch torchvision torchaudio --no-cache-dir
```

#### Permission Issues
```bash
# Fix permission issues
sudo chown -R $(whoami) /usr/local/lib/python3.11/site-packages/
```

#### Homebrew Issues
```bash
# Update Homebrew
brew update && brew upgrade

# Fix Homebrew permissions
sudo chown -R $(whoami) /opt/homebrew/
```

### Model Download Issues

#### Network Issues
```bash
# Download with retry
python scripts/download_models.py --basic --retry 3

# Use specific mirror
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py --basic
```

#### Storage Issues
```bash
# Check disk space
df -h

# Clean unnecessary files
pip cache purge
rm -rf ~/.cache/pip/
```

### Performance Issues

#### Memory Issues
```bash
# Monitor memory usage
htop
vm_stat

# Optimize for low memory
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

#### CPU Issues
```bash
# Check CPU usage
top -o cpu

# Limit CPU threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## Advanced Configuration

### Custom Model Integration

#### Adding Custom Models
```python
# configs/custom_models.yaml
custom_models:
  my_language_model:
    path: "path/to/model"
    type: "language"
    config:
      max_length: 512
      temperature: 0.7
      
  my_vision_model:
    path: "path/to/vision/model"  
    type: "vision"
    config:
      confidence_threshold: 0.8
```

#### Model Configuration
```python
# src/custom_model_loader.py
class CustomModelLoader:
    def load_custom_model(self, model_config):
        # Implementation for custom model loading
        pass
```

### System Integration

#### Service Configuration
```bash
# Create system service (optional)
sudo cp configs/saorsa.service /Library/LaunchDaemons/
sudo launchctl load /Library/LaunchDaemons/saorsa.service
```

#### Logging Configuration
```yaml
# configs/logging.yaml
logging:
  version: 1
  handlers:
    file:
      class: logging.FileHandler
      filename: logs/saorsa.log
      level: INFO
    console:
      class: logging.StreamHandler
      level: DEBUG
```

This software setup guide provides a complete foundation for installing and configuring the Saorsa system. Following these steps ensures a reliable, optimized installation ready for development or production use.