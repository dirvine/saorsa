# Troubleshooting Guide

This comprehensive troubleshooting guide covers common issues and solutions for the Saorsa voice-controlled robot system.

## Quick Diagnostic Tools

### System Status Check
```bash
# Run comprehensive system diagnosis
python src/main_mac.py status --verbose

# Check specific components
python src/main_mac.py test-audio
python src/main_mac.py test-camera
python src/main_mac.py test-robot --port /dev/tty.usbserial-FT1234
```

### Log Analysis
```bash
# View recent logs
tail -f logs/saorsa.log

# Search for errors
grep -i error logs/saorsa.log | tail -20

# Check specific component logs
grep -i "audio\|voice\|whisper" logs/saorsa.log
grep -i "camera\|vision\|detection" logs/saorsa.log
grep -i "robot\|motor\|joint" logs/saorsa.log
```

## Installation and Setup Issues

### Python and Dependencies

#### Problem: Python Version Conflicts
**Symptoms:**
- `ImportError` for required packages
- Version compatibility warnings
- `ModuleNotFoundError` exceptions

**Solutions:**
```bash
# Check Python version
python --version  # Should be 3.11+

# Use specific Python version
python3.11 -m pip install -r requirements.txt

# Create clean virtual environment
python3.11 -m venv saorsa-clean
source saorsa-clean/bin/activate
pip install -r requirements.txt
```

#### Problem: PyTorch Installation Issues
**Symptoms:**
- `torch` module not found
- MPS not available errors
- CUDA-related errors on Apple Silicon

**Solutions:**
```bash
# Clean PyTorch installation
pip uninstall torch torchvision torchaudio
pip cache purge

# Install correct PyTorch for Apple Silicon
pip install torch torchvision torchaudio

# Verify MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

#### Problem: Hugging Face Dependencies
**Symptoms:**
- `transformers` import errors
- Model download failures
- Authentication issues

**Solutions:**
```bash
# Update Hugging Face packages
pip install --upgrade transformers accelerate tokenizers

# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Set authentication (if needed)
huggingface-cli login

# Check model downloads
python scripts/download_models.py --check
```

### macOS Permissions

#### Problem: Microphone Access Denied
**Symptoms:**
- "Microphone permission denied" errors
- No audio input detected
- Whisper initialization fails

**Solutions:**
1. **System Preferences** → **Security & Privacy** → **Privacy** → **Microphone**
2. Add your terminal application (Terminal, iTerm2, etc.)
3. Add Python to the list
4. Restart the application

```bash
# Test microphone access
python -c "
import sounddevice as sd
try:
    devices = sd.query_devices()
    print('Audio devices accessible')
except Exception as e:
    print(f'Audio access error: {e}')
"
```

#### Problem: Camera Access Denied
**Symptoms:**
- "Camera permission denied" errors
- No camera devices found
- Vision system initialization fails

**Solutions:**
1. **System Preferences** → **Security & Privacy** → **Privacy** → **Camera**
2. Add your terminal application and Python
3. Restart applications

```bash
# Test camera access
python -c "
from src.mac_camera_handler import MacCameraHandler
handler = MacCameraHandler()
print('Camera access:', handler.test_camera_access())
"
```

## Hardware Connection Issues

### Robot Connection Problems

#### Problem: Robot Not Found
**Symptoms:**
- "No serial device found" errors
- Connection timeout errors
- `ls /dev/tty.usbserial-*` shows no devices

**Solutions:**
```bash
# Check USB connection
# 1. Verify USB cable is securely connected
# 2. Try different USB port
# 3. Check robot power LED

# Check System Information
System Information → Hardware → USB

# Try different baud rates
python scripts/test_robot_connection.py --port /dev/tty.usbserial-FT1234 --baud 57600
python scripts/test_robot_connection.py --port /dev/tty.usbserial-FT1234 --baud 1000000
```

#### Problem: Robot Connection Timeout
**Symptoms:**
- Connection starts but times out
- Partial communication then failure
- Intermittent connection issues

**Solutions:**
```bash
# Check cable quality
# Replace USB cable with high-quality one

# Verify robot power
# Ensure 12V power supply is connected and LED is on

# Test with simple commands
python -c "
from src.robot_controller_m3 import RobotController, create_default_so101_config
config = create_default_so101_config('/dev/tty.usbserial-FT1234', 'Test')
robot = RobotController(config)
print('Connection result:', robot.connect('/dev/tty.usbserial-FT1234'))
"

# Check for interference
# Move away from WiFi routers, motors, or other electronic devices
```

#### Problem: Motor Errors
**Symptoms:**
- Individual motor errors
- Overheating warnings
- Position errors

**Solutions:**
```bash
# Check motor health
python scripts/diagnose_motors.py /dev/tty.usbserial-FT1234

# Common fixes:
# 1. Power cycle robot
# 2. Check for physical obstructions
# 3. Verify motor mounting
# 4. Check temperature and voltage

# Reset motors
python scripts/reset_motors.py /dev/tty.usbserial-FT1234
```

### Audio Hardware Issues

#### Problem: No Audio Input Detected
**Symptoms:**
- No microphone devices found
- Audio test fails
- Whisper receives no input

**Solutions:**
```bash
# List audio devices
python -c "
import sounddevice as sd
print('Input devices:')
for i, device in enumerate(sd.query_devices()):
    if device['max_input_channels'] > 0:
        print(f'{i}: {device[\"name\"]}')
"

# Test specific device
python scripts/test_audio_device.py --device-id 1

# Check Audio MIDI Setup
# Applications → Utilities → Audio MIDI Setup
# Verify microphone is listed and not muted
```

#### Problem: Poor Voice Recognition
**Symptoms:**
- Commands not recognized
- Low confidence scores
- Frequent misinterpretation

**Solutions:**
```bash
# Improve audio setup:
# 1. Move closer to microphone (12-18 inches)
# 2. Reduce background noise
# 3. Speak clearly and slowly
# 4. Use external microphone

# Adjust Whisper model
# Edit configs/default.yaml:
# audio:
#   whisper_model: "small"  # Try larger model
#   vad_threshold: 0.3      # Lower threshold

# Test different models
python scripts/test_whisper_models.py
```

### Camera Hardware Issues

#### Problem: Camera Not Detected
**Symptoms:**
- No camera devices found
- Vision system fails to start
- Camera permission issues

**Solutions:**
```bash
# Check built-in camera
# Test with Photo Booth app first

# List available cameras
python -c "
from src.mac_camera_handler import MacCameraHandler
handler = MacCameraHandler()
cameras = handler.get_available_cameras()
for cam in cameras:
    print(f'{cam.name}: {cam.device_type}')
"

# Test USB camera
# Try different USB ports
# Check USB hub power delivery

# iPhone Continuity Camera troubleshooting:
# 1. Ensure same Apple ID on both devices
# 2. Enable Bluetooth and WiFi
# 3. Update to latest iOS/macOS
# 4. Reset network settings if needed
```

#### Problem: Poor Object Detection
**Symptoms:**
- Objects not detected
- Many false positives
- Inconsistent detection results

**Solutions:**
```bash
# Improve lighting:
# 1. Add bright, even lighting
# 2. Minimize shadows
# 3. Avoid reflective surfaces
# 4. Use contrasting background

# Adjust detection settings:
# Edit configs/vision.yaml:
# vision:
#   detection_confidence: 0.5  # Lower for more detections
#   detection_confidence: 0.8  # Higher for fewer false positives

# Try different models
python src/object_detector.py --model "yolov8n"
python src/object_detector.py --model "facebook/detr-resnet-101"

# Clean camera lens
# Ensure camera is clean and focused
```

## Software Runtime Issues

### AI Model Issues

#### Problem: Model Loading Failures
**Symptoms:**
- "Model not found" errors
- Download timeouts
- Corrupted model files

**Solutions:**
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/

# Re-download models
python scripts/download_models.py --basic --retry 3

# Check available disk space
df -h

# Use different model
# Edit configs/default.yaml:
# ai_models:
#   primary: "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # Smaller model

# Manual model download
python scripts/download_models.py --model "HuggingFaceTB/SmolLM2-1.7B-Instruct"
```

#### Problem: Out of Memory Errors
**Symptoms:**
- "CUDA out of memory" (even on MPS)
- "MPS out of memory" errors
- System freezes during AI processing

**Solutions:**
```bash
# Use smaller models
python scripts/download_models.py --lightweight

# Enable memory optimization
# Edit configs/mac_m3.yaml:
# model_optimization:
#   low_memory_mode: true
#   fp16_inference: true
#   batch_size: 1

# Monitor memory usage
python scripts/monitor_memory.py

# Clear memory cache
python -c "
import torch
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
"

# Restart system if necessary
```

#### Problem: Slow AI Processing
**Symptoms:**
- Long response times (>10 seconds)
- System becomes unresponsive
- High CPU/GPU usage

**Solutions:**
```bash
# Use faster models
# Edit configs/default.yaml:
# ai_models:
#   primary: "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Enable optimizations
# model_optimization:
#   enable_mps: true
#   torch_compile: true
#   fp16_inference: true

# Monitor performance
python scripts/benchmark_models.py

# Close other applications
# Ensure sufficient system resources
```

### Vision System Issues

#### Problem: Low Frame Rate
**Symptoms:**
- Choppy video feed
- Detection lag
- System performance issues

**Solutions:**
```bash
# Reduce camera resolution
# Edit configs/vision.yaml:
# camera:
#   resolution: [640, 480]  # Lower resolution
#   fps: 15                 # Lower FPS

# Use faster detection model
# vision:
#   object_detection: "yolov8n"  # Fastest model

# Optimize processing
# performance:
#   frame_skip_ratio: 3  # Process every 3rd frame
#   detection_fps: 5     # Lower detection rate

# Monitor system resources
htop  # Check CPU usage
```

#### Problem: Spatial Reference Errors
**Symptoms:**
- "Left" and "right" reversed
- Inaccurate object positioning
- Calibration issues

**Solutions:**
```bash
# Recalibrate camera-robot mapping
python scripts/calibrate_camera_robot.py --robot-port /dev/tty.usbserial-FT1234

# Check camera positioning
# Ensure camera is positioned correctly relative to robot

# Verify workspace setup
# Make sure robot and camera coordinate systems align

# Test spatial references
python scripts/test_spatial_references.py

# Manual calibration
# Edit configs/spatial_calibration.yaml with manual adjustments
```

### Multimodal Integration Issues

#### Problem: Voice-Vision Sync Issues
**Symptoms:**
- Commands refer to wrong objects
- Temporal misalignment
- Context confusion

**Solutions:**
```bash
# Adjust timing parameters
# Edit configs/multimodal.yaml:
# timing:
#   voice_vision_sync_window: 2.0  # Seconds
#   context_retention_time: 30.0   # Seconds

# Improve command clarity
# Use more specific language: "pick up the red block on the left"
# Avoid ambiguous references: "that thing over there"

# Reset context
python -c "
from src.context_manager import ContextManager
context = ContextManager()
# Context automatically resets
"
```

## Performance Optimization

### System Performance

#### Problem: High CPU Usage
**Symptoms:**
- System slowdown
- Fan noise
- Overheating

**Solutions:**
```bash
# Monitor processes
top -o cpu

# Optimize model settings
# Reduce model size
# Lower processing frequencies
# Enable efficient inference modes

# Close unnecessary applications
# Quit other CPU-intensive apps
# Disable background processes

# Check Activity Monitor
# Look for runaway processes
```

#### Problem: High Memory Usage
**Symptoms:**
- System swapping
- Out of memory errors
- Slow performance

**Solutions:**
```bash
# Monitor memory usage
vm_stat
htop

# Use memory-efficient models
python scripts/download_models.py --lightweight

# Enable memory optimization
# Edit configs/mac_m3.yaml:
# memory_optimization:
#   low_memory_mode: true
#   model_offloading: true

# Restart application periodically
# Memory leaks may require periodic restarts
```

### Network and Connectivity

#### Problem: Model Download Issues
**Symptoms:**
- Download timeouts
- Connection errors
- Slow download speeds

**Solutions:**
```bash
# Check internet connection
ping 8.8.8.8

# Use alternative mirror
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_models.py --basic

# Download with retry
python scripts/download_models.py --basic --retry 5

# Use wget/curl for manual download
wget https://huggingface.co/model_path/resolve/main/pytorch_model.bin
```

## Error Code Reference

### Common Error Codes

#### Robot Controller Errors
```
RC_001: Connection timeout
RC_002: Invalid motor ID
RC_003: Position out of range
RC_004: Safety violation
RC_005: Communication error
RC_006: Motor overheating
RC_007: Power supply error
```

#### Audio System Errors
```
AU_001: Microphone not found
AU_002: Permission denied
AU_003: Whisper model not loaded
AU_004: Audio processing timeout
AU_005: VAD threshold error
```

#### Vision System Errors
```
VS_001: Camera not detected
VS_002: Model loading failed
VS_003: Detection timeout
VS_004: Calibration error
VS_005: Frame processing error
```

#### AI Processing Errors
```
AI_001: Model not found
AI_002: Out of memory
AI_003: Processing timeout
AI_004: Context error
AI_005: Invalid command format
```

## Advanced Troubleshooting

### Debug Mode

#### Enable Debug Logging
```bash
# Run with debug logging
python src/main_mac.py run --verbose --log-level DEBUG

# Enable component-specific debugging
export SAORSE_DEBUG_AUDIO=1
export SAORSE_DEBUG_VISION=1
export SAORSE_DEBUG_ROBOT=1
```

#### Debug Configuration
```yaml
# configs/debug.yaml
logging:
  level: DEBUG
  components:
    audio: DEBUG
    vision: DEBUG
    robot: DEBUG
    ai: DEBUG
    
debug:
  save_frames: true
  save_audio: true
  detailed_timing: true
  memory_profiling: true
```

### Profiling and Diagnostics

#### Performance Profiling
```bash
# Profile AI model performance
python scripts/profile_ai_models.py

# Profile vision system
python scripts/profile_vision.py

# System resource monitoring
python scripts/monitor_system_resources.py --duration 300
```

#### Memory Profiling
```bash
# Memory usage profiling
python -m memory_profiler src/main_mac.py run --duration 60

# Track memory leaks
python scripts/track_memory_leaks.py
```

### Recovery Procedures

#### System Recovery
```bash
# Emergency stop all processes
pkill -f saorsa
pkill -f python

# Reset robot to safe state
python scripts/emergency_reset.py --port /dev/tty.usbserial-FT1234

# Clear all caches
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/pip/
rm -rf logs/*

# Restart with clean configuration
python src/main_mac.py run --config configs/minimal.yaml
```

#### Data Recovery
```bash
# Backup current configuration
cp -r configs/ configs_backup_$(date +%Y%m%d)/

# Restore default configuration
git checkout configs/

# Recover from logs
python scripts/recover_from_logs.py --log-file logs/saorsa.log
```

## Getting Help

### Log Collection for Support

#### Collect Diagnostic Information
```bash
# Generate support package
python scripts/generate_support_package.py

# This creates a zip file with:
# - System information
# - Configuration files
# - Recent logs
# - Error reports
# - Performance metrics
```

#### Manual Log Collection
```bash
# System information
system_profiler SPHardwareDataType > system_info.txt
sw_vers > os_version.txt

# Python environment
pip list > python_packages.txt
python --version > python_version.txt

# Recent logs
tail -1000 logs/saorsa.log > recent_logs.txt

# Configuration
cp -r configs/ config_backup/
```

### Common Support Questions

#### Before Asking for Help

1. **Check system requirements**: Ensure macOS 13.0+, Apple Silicon, 16GB+ RAM
2. **Verify installation**: Run `python src/main_mac.py status`
3. **Check logs**: Look for recent errors in `logs/saorsa.log`
4. **Try basic test**: Run `python src/main_mac.py test-audio`
5. **Update software**: Ensure latest version from repository

#### Information to Include

- **System**: macOS version, Mac model, RAM amount
- **Error message**: Complete error text and stack trace
- **Reproduction steps**: Exact commands and sequence
- **Configuration**: Relevant configuration file contents
- **Logs**: Recent log entries around the time of error

This troubleshooting guide should help resolve most common issues with the Saorsa robot control system. For additional support, include the diagnostic information described above when reporting issues.