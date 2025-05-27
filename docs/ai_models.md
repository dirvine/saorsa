# AI Models Guide

This guide covers the artificial intelligence models used in the Saorse robot control system, including selection criteria, performance characteristics, and optimization strategies for Apple Silicon.

## Model Architecture Overview

The Saorse system uses two main categories of AI models:

1. **Language Models**: For natural language understanding and command interpretation
2. **Vision Models**: For computer vision and object detection (Phase 3)

Both model types are optimized for local execution on Apple Silicon hardware using Metal Performance Shaders (MPS).

## Language Models (Phase 2)

### Model Categories

#### Lightweight Models (8-16GB RAM)
Optimized for systems with limited memory while maintaining good performance.

**SmolLM2-1.7B-Instruct**
```yaml
model: "HuggingFaceTB/SmolLM2-1.7B-Instruct"
size: ~3.4GB
parameters: 1.7B
performance: Fast inference, good for basic commands
use_case: Entry-level systems, basic robot control
```

**Phi-3.5-mini-instruct**  
```yaml
model: "microsoft/Phi-3.5-mini-instruct"
size: ~7.6GB
parameters: 3.8B
performance: Excellent quality-to-size ratio
use_case: Balanced performance and memory usage
```

#### Balanced Models (16-32GB RAM)
Optimal balance of performance and resource usage for most users.

**Qwen2.5-3B-Instruct**
```yaml
model: "Qwen/Qwen2.5-3B-Instruct"
size: ~6.0GB
parameters: 3B
performance: High-quality responses, fast inference
use_case: General-purpose robot control
```

**Qwen2.5-7B-Instruct**
```yaml
model: "Qwen/Qwen2.5-7B-Instruct"
size: ~14GB
parameters: 7B
performance: Excellent understanding, moderate speed
use_case: Complex task planning and execution
```

#### High-Performance Models (32GB+ RAM)
Maximum capability models for advanced applications.

**Qwen2.5-14B-Instruct**
```yaml
model: "Qwen/Qwen2.5-14B-Instruct"
size: ~28GB
parameters: 14B
performance: Best-in-class understanding
use_case: Complex reasoning, advanced robotics
```

**Llama-3.2-3B-Instruct**
```yaml
model: "meta-llama/Llama-3.2-3B-Instruct"
size: ~6.0GB
parameters: 3B
performance: Strong reasoning capabilities
use_case: Research and development
```

### Model Selection Guidelines

#### By System Specifications

**8-16GB RAM Systems**
```yaml
recommended_models:
  primary: "HuggingFaceTB/SmolLM2-1.7B-Instruct"
  fallback: "microsoft/DialoGPT-medium"
  whisper: "base"

optimization:
  fp16_inference: true
  low_memory_mode: true
  batch_size: 1
```

**16-32GB RAM Systems**
```yaml
recommended_models:
  primary: "Qwen/Qwen2.5-3B-Instruct"
  fallback: "microsoft/Phi-3.5-mini-instruct"
  whisper: "small"

optimization:
  fp16_inference: true
  enable_mps: true
  batch_size: 1
```

**32GB+ RAM Systems**
```yaml
recommended_models:
  primary: "Qwen/Qwen2.5-14B-Instruct"
  fallback: "meta-llama/Llama-3.2-3B-Instruct"
  whisper: "medium"

optimization:
  fp16_inference: false
  enable_mps: true
  batch_size: 2
```

#### By Use Case

**Basic Robot Control**
- Focus: Reliability and speed
- Model: SmolLM2-1.7B-Instruct
- Features: Basic command interpretation, simple task execution

**Advanced Manipulation**
- Focus: Understanding complex instructions
- Model: Qwen2.5-7B-Instruct
- Features: Context awareness, multi-step planning

**Research and Development**
- Focus: Maximum capability
- Model: Qwen2.5-14B-Instruct
- Features: Advanced reasoning, complex task planning

### Performance Characteristics

#### Inference Speed Comparison (M3 Max)
```
Model                          | Tokens/sec | Latency (avg)
SmolLM2-1.7B-Instruct         | 45-60      | 1.2s
Phi-3.5-mini-instruct         | 35-50      | 1.8s
Qwen2.5-3B-Instruct           | 30-45      | 2.1s
Qwen2.5-7B-Instruct           | 20-35      | 3.2s
Qwen2.5-14B-Instruct          | 12-20      | 5.1s
```

#### Memory Usage (FP16)
```
Model                          | VRAM Usage | System RAM
SmolLM2-1.7B-Instruct         | 3.4GB      | 2.1GB
Phi-3.5-mini-instruct         | 7.6GB      | 4.2GB
Qwen2.5-3B-Instruct           | 6.0GB      | 3.8GB
Qwen2.5-7B-Instruct           | 14GB       | 8.5GB
Qwen2.5-14B-Instruct          | 28GB       | 16GB
```

## Vision Models (Phase 3)

### Object Detection Models

#### DETR (Detection Transformer) Models
State-of-the-art transformer-based object detection.

**DETR-ResNet-50**
```yaml
model: "facebook/detr-resnet-50"
size: ~159MB
backbone: ResNet-50
performance: Good balance of speed and accuracy
use_case: General object detection
```

**DETR-ResNet-101**
```yaml
model: "facebook/detr-resnet-101"
size: ~232MB
backbone: ResNet-101
performance: Higher accuracy, slower inference
use_case: High-precision applications
```

#### YOLO Models
Real-time object detection with excellent speed.

**YOLOv8 Variants**
```yaml
yolov8n:  # Nano
  size: ~6MB
  speed: >100 FPS
  accuracy: Good
  use_case: Real-time applications

yolov8s:  # Small
  size: ~22MB
  speed: ~80 FPS
  accuracy: Better
  use_case: Balanced performance

yolov8m:  # Medium
  size: ~52MB
  speed: ~50 FPS
  accuracy: High
  use_case: Accuracy-focused applications

yolov8l:  # Large
  size: ~87MB
  speed: ~30 FPS
  accuracy: Very High
  use_case: Maximum accuracy
```

#### Specialized Models

**Table Transformer**
```yaml
model: "microsoft/table-transformer-object-detection"
size: ~221MB
specialty: Table and document detection
use_case: Structured workspace organization
```

### Vision Model Selection

#### By Performance Requirements

**Real-Time Operation (>30 FPS)**
```yaml
recommended:
  primary: "yolov8n"
  fallback: "yolov8s"
  
optimization:
  input_size: [640, 640]
  confidence_threshold: 0.5
  nms_threshold: 0.4
```

**High Accuracy (<15 FPS acceptable)**
```yaml
recommended:
  primary: "facebook/detr-resnet-101"
  fallback: "yolov8l"
  
optimization:
  input_size: [800, 600]
  confidence_threshold: 0.7
  nms_threshold: 0.5
```

**Balanced Performance (~20 FPS)**
```yaml
recommended:
  primary: "facebook/detr-resnet-50"
  fallback: "yolov8m"
  
optimization:
  input_size: [800, 600]
  confidence_threshold: 0.6
  nms_threshold: 0.5
```

## Model Installation and Management

### Downloading Models

#### Basic Installation
```bash
# Download essential models
python scripts/download_models.py --basic

# Download specific model
python scripts/download_models.py --model "Qwen/Qwen2.5-3B-Instruct"

# Download all models for complete functionality
python scripts/download_models.py --all
```

#### Selective Installation
```bash
# Language models only
python scripts/download_models.py --language-only

# Vision models only
python scripts/download_models.py --vision-only

# Lightweight models for limited systems
python scripts/download_models.py --lightweight

# High-performance models for powerful systems
python scripts/download_models.py --high-performance
```

#### Model Verification
```bash
# Check downloaded models
python scripts/download_models.py --check

# List available models
python scripts/download_models.py --list-available

# Test specific model
python scripts/test_model.py --model "Qwen/Qwen2.5-3B-Instruct"
```

### Model Configuration

#### Language Model Configuration
```yaml
# configs/ai_models.yaml
language_models:
  primary:
    name: "Qwen/Qwen2.5-3B-Instruct"
    max_length: 512
    temperature: 0.7
    top_p: 0.9
    do_sample: true
    
  fallback:
    name: "microsoft/Phi-3.5-mini-instruct"
    max_length: 256
    temperature: 0.8
    
optimization:
  enable_mps: true
  fp16_inference: true
  torch_compile: false  # Disable if causing issues
```

#### Vision Model Configuration
```yaml
# configs/vision_models.yaml
vision_models:
  object_detection:
    name: "facebook/detr-resnet-50"
    confidence_threshold: 0.6
    nms_threshold: 0.5
    max_detections: 100
    
  tracking:
    enable: true
    max_age: 30
    min_hits: 3
    
optimization:
  input_size: [800, 600]
  batch_size: 1
  enable_mps: true
```

## Apple Silicon Optimization

### Metal Performance Shaders (MPS)

#### MPS Configuration
```python
# Verify MPS availability
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Configure MPS device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
```

#### MPS Optimization Settings
```yaml
mps_optimization:
  high_watermark_ratio: 0.8     # Memory usage limit
  low_watermark_ratio: 0.6      # Memory cleanup threshold
  enable_fallback: true         # Fallback to CPU if needed
  memory_fraction: 0.8          # GPU memory allocation
```

### Memory Optimization

#### Memory-Efficient Loading
```python
# Load model with reduced precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,    # Use FP16
    low_cpu_mem_usage=True,       # Reduce CPU memory during loading
    device_map="auto"             # Automatic device placement
)
```

#### Memory Monitoring
```python
# Monitor memory usage
import torch
if torch.backends.mps.is_available():
    print(f"MPS memory allocated: {torch.mps.current_allocated_memory()/1024**3:.2f} GB")
    print(f"MPS memory cached: {torch.mps.driver_allocated_memory()/1024**3:.2f} GB")
```

### Performance Tuning

#### Inference Optimization
```python
# Model compilation for speed
model = torch.compile(model, mode="reduce-overhead")

# Disable gradient computation
torch.set_grad_enabled(False)

# Use faster attention implementation
torch.backends.cuda.enable_flash_sdp(False)  # Disable for MPS
```

#### Batch Processing
```python
# Optimal batch sizes for different models
batch_sizes = {
    "SmolLM2-1.7B": 4,
    "Qwen2.5-3B": 2,
    "Qwen2.5-7B": 1,
    "Qwen2.5-14B": 1
}
```

## Model Evaluation and Benchmarking

### Performance Metrics

#### Language Model Evaluation
```bash
# Run language model benchmarks
python scripts/benchmark_language_models.py

# Test specific model performance
python scripts/test_model_performance.py --model "Qwen/Qwen2.5-3B-Instruct"

# Evaluate command understanding accuracy
python scripts/evaluate_command_accuracy.py
```

#### Vision Model Evaluation
```bash
# Run vision model benchmarks
python scripts/benchmark_vision_models.py

# Test detection accuracy
python scripts/test_detection_accuracy.py --model "facebook/detr-resnet-50"

# Measure inference speed
python scripts/measure_vision_speed.py
```

### Quality Assessment

#### Language Model Quality
- **Response Relevance**: How well the model understands robot commands
- **Context Awareness**: Ability to maintain conversation context
- **Safety**: Avoidance of harmful or dangerous commands
- **Consistency**: Reproducible responses for similar inputs

#### Vision Model Quality
- **Detection Accuracy**: Precision and recall for object detection
- **False Positive Rate**: Frequency of incorrect detections
- **Real-time Performance**: Consistent frame rates during operation
- **Robustness**: Performance under varying lighting conditions

## Troubleshooting Model Issues

### Common Problems

#### Model Loading Failures
```bash
# Check model cache
ls ~/.cache/huggingface/transformers/

# Clear cache and re-download
rm -rf ~/.cache/huggingface/
python scripts/download_models.py --model "model_name"

# Check available disk space
df -h
```

#### Memory Issues
```bash
# Monitor memory usage
htop
vm_stat

# Reduce model size
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Use smaller models
python scripts/download_models.py --lightweight
```

#### Performance Issues
```bash
# Check MPS availability
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# Monitor GPU usage
sudo powermetrics --samplers gpu_power -n 1

# Profile model performance
python scripts/profile_model.py --model "model_name"
```

### Optimization Strategies

#### For Speed
1. Use smaller models (SmolLM2, YOLOv8n)
2. Enable FP16 inference
3. Reduce input sequence length
4. Use model compilation
5. Optimize batch sizes

#### For Accuracy
1. Use larger models (Qwen2.5-14B, DETR-ResNet-101)
2. Increase confidence thresholds
3. Use ensemble methods
4. Fine-tune on specific tasks
5. Implement post-processing filters

#### For Memory Efficiency
1. Enable model offloading
2. Use gradient checkpointing
3. Reduce batch sizes
4. Clear cache regularly
5. Use quantized models

## Custom Model Integration

### Adding New Models

#### Language Models
```python
# Custom model configuration
custom_language_model = {
    "name": "custom/my-robot-model",
    "type": "causal_lm",
    "tokenizer": "custom/my-robot-tokenizer",
    "config": {
        "max_length": 512,
        "temperature": 0.7
    }
}
```

#### Vision Models
```python
# Custom vision model configuration
custom_vision_model = {
    "name": "custom/my-detection-model",
    "type": "object_detection",
    "preprocessor": "custom/my-preprocessor",
    "config": {
        "confidence_threshold": 0.8,
        "input_size": [640, 640]
    }
}
```

### Model Fine-tuning

#### Preparing Training Data
```python
# Robot command dataset format
training_data = {
    "commands": [
        {
            "input": "pick up the red block",
            "output": "execute_pickup(object='red_block')",
            "context": "workspace_scan"
        }
    ]
}
```

#### Fine-tuning Process
```bash
# Fine-tune language model for robot commands
python scripts/fine_tune_language_model.py \
    --base_model "Qwen/Qwen2.5-3B-Instruct" \
    --dataset "data/robot_commands.json" \
    --output_dir "models/custom_robot_model"

# Fine-tune vision model for specific objects
python scripts/fine_tune_vision_model.py \
    --base_model "facebook/detr-resnet-50" \
    --dataset "data/robot_objects.coco" \
    --output_dir "models/custom_detection_model"
```

This comprehensive AI models guide provides the foundation for understanding, selecting, and optimizing artificial intelligence models within the Saorse robot control system.