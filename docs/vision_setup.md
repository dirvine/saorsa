# Vision System Setup Guide

This guide covers the complete setup and configuration of the computer vision system for Saorsa Phase 3 multimodal robot control.

## Overview

The Saorsa vision system provides:
- Real-time object detection and tracking
- Spatial reference resolution for voice commands
- Visual feedback with detection overlays
- Support for multiple camera types including iPhone Continuity Camera

## Camera Hardware Setup

### Built-in Cameras

#### MacBook Pro/Air
- **Camera**: FaceTime HD (720p) or 1080p FaceTime HD
- **Quality**: Good for basic object detection
- **Positioning**: Adjust screen angle for optimal workspace view
- **Limitations**: Fixed position, may require workspace adjustment

#### iMac
- **Camera**: FaceTime HD or 12MP Ultra Wide (newer models)
- **Quality**: Excellent for object detection
- **Positioning**: Optimal height and angle for workspace
- **Benefits**: Stable position, good lighting angle

#### Mac Studio/Mac Pro
- **Camera**: External camera required
- **Recommendation**: High-quality USB or capture card camera
- **Flexibility**: Can position optimally for workspace

### External USB Cameras

#### Recommended USB Cameras

**Logitech C920/C922 Pro**
```yaml
specifications:
  resolution: 1080p @ 30fps
  field_of_view: 78°
  autofocus: Yes
  price_range: $70-100
  pros: Reliable, good software support, auto-exposure
  cons: Average low-light performance
```

**Logitech BRIO**
```yaml
specifications:
  resolution: 4K @ 30fps, 1080p @ 60fps
  field_of_view: 90°
  autofocus: Yes
  price_range: $150-200
  pros: Excellent quality, high frame rates, wide FOV
  cons: Higher cost, requires good lighting
```

**Razer Kiyo**
```yaml
specifications:
  resolution: 1080p @ 30fps
  field_of_view: 81.6°
  ring_light: Built-in
  price_range: $100-130
  pros: Built-in lighting, good for darker environments
  cons: Ring light may cause reflections
```

#### Camera Positioning
```
Optimal Camera Setup:

     [Camera]
        |
        | 2-4 feet
        |
        v
   [Workspace]
   [Robot Base]
```

**Height**: 18-36 inches above workspace
**Angle**: 15-30 degrees downward
**Distance**: 2-4 feet from robot workspace
**Lighting**: Even lighting across workspace

### iPhone Continuity Camera

#### Requirements
- **iPhone**: iPhone 8 or later
- **iOS**: iOS 16.0 or later
- **macOS**: macOS Ventura (13.0) or later
- **Apple ID**: Same Apple ID on both devices
- **Network**: Same WiFi network and Bluetooth enabled

#### Setup Process
1. **Update Devices**: Ensure both iPhone and Mac are updated
2. **Apple ID**: Sign in with same Apple ID on both devices
3. **Proximity**: Keep iPhone within Bluetooth range of Mac
4. **Automatic Detection**: iPhone should appear as camera option

#### Benefits
- **Quality**: Often superior to built-in Mac cameras
- **Flexibility**: Can position iPhone optimally
- **Features**: Advanced computational photography
- **Convenience**: Wireless operation

#### iPhone Positioning
```bash
# iPhone mount options:
# - Tripod with phone mount
# - Desk arm with phone holder
# - Wall mount for fixed position
# - Magnetic mount for quick adjustment
```

### Professional Camera Setup

#### DSLR/Mirrorless Cameras
For highest quality applications, professional cameras can be integrated via capture cards.

**Recommended Setup**:
```yaml
camera: Canon EOS R5, Sony A7 IV, Fujifilm X-T5
capture_card: Elgato Cam Link 4K, Blackmagic Design
connection: HDMI to USB-C capture
benefits: 
  - Maximum image quality
  - Professional autofocus
  - Excellent low-light performance
  - Interchangeable lenses
```

## Software Configuration

### Camera Permissions

#### macOS Privacy Settings
1. **System Preferences** → **Security & Privacy** → **Privacy**
2. **Camera** → Add Terminal, Python, and your development environment
3. **Restart** applications that need camera access

#### Verify Permissions
```bash
# Test camera access
python src/main_mac.py test-camera

# List available cameras
python -c "
from src.mac_camera_handler import MacCameraHandler
handler = MacCameraHandler()
cameras = handler.get_available_cameras()
for i, cam in enumerate(cameras):
    print(f'{i}: {cam.name} ({cam.device_type})')
"
```

### Camera Configuration

#### Basic Camera Settings
```yaml
# configs/vision.yaml
camera:
  resolution: [1280, 720]          # 720p for balanced performance
  fps: 30                          # 30 FPS for smooth operation
  buffer_size: 5                   # Frame buffer size
  enable_continuity_camera: true   # iPhone camera support
  enable_usb_cameras: true         # External USB cameras
  enable_builtin_camera: true      # Built-in Mac cameras
```

#### Advanced Camera Settings
```yaml
camera_advanced:
  auto_exposure: true              # Automatic exposure adjustment
  auto_focus: true                 # Automatic focus adjustment
  exposure_compensation: 0         # Manual exposure adjustment
  white_balance: "auto"            # Color temperature adjustment
  stabilization: true              # Image stabilization (if supported)
```

#### Performance Optimization
```yaml
performance:
  frame_skip_ratio: 2              # Process every Nth frame for detection
  detection_fps: 10                # Target FPS for object detection
  max_processing_fps: 30           # Maximum overall processing rate
  enable_threading: true           # Multi-threaded frame processing
```

### Computer Vision Models

#### Object Detection Models

**Default Configuration**
```yaml
# configs/vision_models.yaml
object_detection:
  primary_model: "facebook/detr-resnet-50"
  confidence_threshold: 0.6
  nms_threshold: 0.5
  max_detections: 100
  input_size: [800, 600]
```

**Performance-Optimized Configuration**
```yaml
object_detection_fast:
  primary_model: "yolov8n"         # Fastest YOLO variant
  confidence_threshold: 0.5
  nms_threshold: 0.4
  max_detections: 50
  input_size: [640, 640]
```

**Accuracy-Optimized Configuration**
```yaml
object_detection_accurate:
  primary_model: "facebook/detr-resnet-101"
  confidence_threshold: 0.7
  nms_threshold: 0.5
  max_detections: 200
  input_size: [1024, 768]
```

#### Model Download and Setup
```bash
# Download vision models
python scripts/download_models.py --vision-only

# Download specific models
python scripts/download_models.py --model "facebook/detr-resnet-50"
python scripts/download_models.py --model "yolov8n"

# Verify model installation
python scripts/download_models.py --check
```

## Testing and Calibration

### Camera Testing

#### Basic Camera Test
```bash
# Test camera access and basic functionality
python src/main_mac.py test-camera
```

#### Camera Quality Assessment
```bash
# Capture test frames and analyze quality
python scripts/test_camera_quality.py --resolution 1280x720 --duration 30

# Test different cameras
python scripts/compare_cameras.py
```

#### Frame Rate Testing
```bash
# Measure actual frame rates
python scripts/test_camera_fps.py --target-fps 30

# Benchmark camera performance
python scripts/benchmark_camera.py
```

### Object Detection Testing

#### Basic Detection Test
```bash
# Test object detection with camera feed
python src/main_mac.py demo-vision

# Test with specific model
python src/object_detector.py --model "facebook/detr-resnet-50"
```

#### Detection Accuracy Testing
```bash
# Test detection accuracy with known objects
python scripts/test_detection_accuracy.py --objects "cup,bottle,block"

# Measure detection performance
python scripts/benchmark_detection.py --duration 60
```

#### Real-time Performance Testing
```bash
# Test multimodal integration performance
python scripts/test_multimodal_performance.py --duration 120

# Monitor system resources during operation
python scripts/monitor_vision_performance.py
```

### Workspace Calibration

#### Camera-to-Robot Calibration

**Calibration Process**:
1. Place calibration markers in robot workspace
2. Record marker positions in robot coordinates
3. Capture images with camera
4. Calculate transformation matrix

```bash
# Run calibration procedure
python scripts/calibrate_camera_robot.py --robot-port /dev/tty.usbserial-FT1234

# Verify calibration accuracy
python scripts/verify_calibration.py
```

#### Spatial Reference Calibration
```bash
# Calibrate spatial references (left, right, center, etc.)
python scripts/calibrate_spatial_references.py

# Test spatial reference accuracy
python scripts/test_spatial_references.py
```

## Visual Feedback Configuration

### Overlay Settings

#### Basic Overlay Configuration
```yaml
# configs/visual_feedback.yaml
overlays:
  show_detections: true            # Show detection bounding boxes
  show_robot_status: true          # Show robot information
  show_workspace_bounds: true      # Show workspace boundaries
  show_safety_zones: true          # Show safety zone indicators
  show_trajectory: true            # Show robot trajectory
  fps_display: true                # Show FPS counter
```

#### Overlay Appearance
```yaml
overlay_appearance:
  detection_box_thickness: 2       # Bounding box line thickness
  text_scale: 0.6                  # Text size scaling
  overlay_alpha: 0.7               # Overlay transparency
  colors:
    detection_box: [0, 255, 0]     # Green detection boxes
    robot_status: [255, 255, 255]  # White status text
    workspace_bounds: [0, 0, 255]  # Blue workspace boundaries
```

#### Advanced Overlays
```yaml
advanced_overlays:
  show_3d_projection: true         # Project 3D coordinates
  show_object_tracking: true       # Show object tracking IDs
  show_confidence_scores: true     # Show detection confidence
  show_spatial_grid: true          # Show coordinate grid
  show_depth_estimation: true      # Show estimated object depth
```

### Display Output

#### Live Display Options
```bash
# Display live camera feed with overlays
python scripts/display_live_feed.py --show-overlays

# Save annotated frames
python scripts/save_annotated_frames.py --duration 60 --output-dir frames/
```

#### Recording and Analysis
```bash
# Record session with overlays
python scripts/record_session.py --duration 300 --output session.mp4

# Analyze recorded session
python scripts/analyze_session.py --input session.mp4
```

## Lighting Optimization

### Lighting Requirements

#### Optimal Lighting Conditions
- **Brightness**: Bright, even lighting across workspace
- **Shadows**: Minimize harsh shadows
- **Reflection**: Avoid reflective surfaces that cause glare
- **Consistency**: Stable lighting without flicker

#### Lighting Setup Recommendations
```
Lighting Configuration:

[Light 1]     [Light 2]
    \           /
     \         /
      \       /
    [Workspace]
    [  Robot  ]
```

**LED Panel Lights**: Provide even, adjustable lighting
**Desk Lamps**: Position to minimize shadows
**Ring Lights**: Good for eliminating shadows (watch for reflections)
**Natural Light**: Supplement with artificial light for consistency

### Color and Contrast Optimization

#### Background Setup
- **Color**: Neutral colors (white, gray, beige)
- **Texture**: Minimal texture to avoid detection interference
- **Contrast**: High contrast between objects and background
- **Cleanliness**: Keep workspace clean and uncluttered

#### Object Visibility
```yaml
object_optimization:
  high_contrast_objects: true      # Use objects with distinct colors
  avoid_similar_colors: true       # Separate similar-colored objects
  good_lighting: true              # Ensure even illumination
  minimal_shadows: true            # Position lights to reduce shadows
```

## Troubleshooting Vision Issues

### Common Problems

#### Camera Not Detected
**Symptoms**: No camera devices found
**Solutions**:
```bash
# Check camera permissions
System Preferences → Security & Privacy → Camera

# Verify USB connection
ls /dev/video*  # Linux style check
system_profiler SPUSBDataType | grep -i camera  # macOS

# Test with built-in Photo Booth app
open -a "Photo Booth"
```

#### Poor Object Detection
**Symptoms**: Objects not detected or false positives
**Solutions**:
```bash
# Improve lighting
# Increase contrast between objects and background
# Clean camera lens
# Adjust detection confidence threshold

# Test with different models
python src/object_detector.py --model "yolov8n"
python src/object_detector.py --model "facebook/detr-resnet-101"
```

#### Low Frame Rate
**Symptoms**: Choppy video or slow detection
**Solutions**:
```bash
# Reduce camera resolution
# Lower detection FPS target
# Use faster detection model
# Close other applications using camera/CPU

# Monitor performance
python scripts/monitor_vision_performance.py
```

#### Spatial Reference Errors
**Symptoms**: "Left" and "right" commands reversed or inaccurate
**Solutions**:
```bash
# Recalibrate camera-robot transformation
python scripts/calibrate_camera_robot.py

# Check camera positioning
# Verify workspace setup matches configuration
```

### Performance Optimization

#### For Speed
1. **Use faster models**: YOLOv8n instead of DETR
2. **Reduce resolution**: 640x480 instead of 1280x720
3. **Lower detection FPS**: 5-10 FPS instead of 30 FPS
4. **Skip frames**: Process every 2nd or 3rd frame
5. **Optimize confidence**: Lower threshold for fewer computations

#### For Accuracy
1. **Use accurate models**: DETR-ResNet-101
2. **Increase resolution**: 1920x1080 for fine details
3. **Higher confidence**: Stricter detection thresholds
4. **Better lighting**: Professional lighting setup
5. **Camera quality**: Use high-quality external camera

#### For Reliability
1. **Stable lighting**: Consistent illumination
2. **Fixed camera**: Avoid camera movement
3. **Clean workspace**: Minimize clutter and distractions
4. **Regular calibration**: Periodic recalibration
5. **Backup cameras**: Multiple camera options configured

## Advanced Configuration

### Multi-Camera Setup

#### Dual Camera Configuration
```yaml
# configs/multi_camera.yaml
cameras:
  primary:
    device_id: "camera_1"
    position: "overhead"
    resolution: [1280, 720]
    
  secondary:
    device_id: "camera_2"
    position: "side_view"
    resolution: [640, 480]
    
processing:
  fusion_method: "weighted_average"
  primary_weight: 0.7
  secondary_weight: 0.3
```

### Custom Object Training

#### Training Data Collection
```bash
# Collect training images
python scripts/collect_training_data.py --objects "custom_block,special_tool" --count 100

# Annotate images for training
python scripts/annotate_images.py --input-dir training_images/ --output annotations.json
```

#### Model Fine-tuning
```bash
# Fine-tune detection model for custom objects
python scripts/fine_tune_detection.py \
    --base-model "facebook/detr-resnet-50" \
    --training-data annotations.json \
    --output-model "models/custom_detector"
```

### Integration with External Systems

#### ROS Integration
```python
# Example ROS vision node
class SaorsaVisionNode:
    def __init__(self):
        self.vision_system = ObjectDetector()
        self.publisher = rospy.Publisher('/saorsa/detections', DetectionArray)
        
    def process_frame(self, frame):
        detections = self.vision_system.detect_objects(frame)
        self.publish_detections(detections)
```

#### API Integration
```python
# REST API for vision system
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    frame = request.get_image()
    detections = vision_system.detect_objects(frame)
    return jsonify(detections)
```

This comprehensive vision setup guide provides all the necessary information for configuring and optimizing the computer vision system in Saorsa for reliable multimodal robot control.