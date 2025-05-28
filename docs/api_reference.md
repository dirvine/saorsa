# API Reference

This document provides comprehensive API documentation for the Saorsa robot control system, covering all modules, classes, and functions.

## Core Modules

### RobotController

#### Class: `RobotController`

Main interface for controlling SO-101 robot arms.

```python
class RobotController:
    def __init__(self, leader_config: RobotConfig, follower_config: Optional[RobotConfig] = None)
```

**Parameters:**
- `leader_config`: Configuration for primary robot arm
- `follower_config`: Optional configuration for secondary robot arm

**Methods:**

##### `connect(leader_port: str, follower_port: Optional[str] = None) -> bool`
Establish connection to robot arm(s).

```python
# Connect to single robot
robot.connect("/dev/tty.usbserial-FT1234")

# Connect to dual robots
robot.connect("/dev/tty.usbserial-FT1234", "/dev/tty.usbserial-FT5678")
```

**Returns:** `bool` - True if connection successful

##### `disconnect() -> None`
Disconnect from all robot arms.

##### `move_joint(robot_id: str, joint_id: int, position: float, speed: int = 50) -> bool`
Move specific joint to target position.

**Parameters:**
- `robot_id`: "leader" or "follower"
- `joint_id`: Joint number (1-6)
- `position`: Target position in degrees
- `speed`: Movement speed (0-100)

##### `move_joints(robot_id: str, positions: Dict[int, float], speed: int = 50) -> bool`
Move multiple joints simultaneously.

**Parameters:**
- `robot_id`: "leader" or "follower"
- `positions`: Dictionary mapping joint IDs to target positions
- `speed`: Movement speed (0-100)

##### `set_gripper(robot_id: str, value: int) -> bool`
Control gripper position.

**Parameters:**
- `robot_id`: "leader" or "follower"
- `value`: Gripper position (0=closed, 100=open)

##### `home_position(robot_id: str = "leader") -> bool`
Move robot to home position (all joints at 0 degrees).

##### `emergency_stop() -> None`
Immediately stop all robot movement.

##### `get_robot_state(robot_id: str) -> RobotState`
Get current robot state including joint positions and status.

**Returns:** `RobotState` object with current robot information

#### Class: `RobotConfig`

Configuration settings for robot arms.

```python
@dataclass
class RobotConfig:
    port: str                           # Serial port path
    name: str                           # Robot identifier
    baud_rate: int = 1000000           # Communication baud rate
    motor_ids: List[int] = field(default_factory=lambda: [1,2,3,4,5,6])
    joint_limits: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    safety_limits: Dict[str, float] = field(default_factory=dict)
```

#### Class: `RobotState`

Current state information for a robot arm.

```python
@dataclass
class RobotState:
    robot_id: str
    is_connected: bool
    joint_positions: Dict[int, float]
    joint_velocities: Dict[int, float]
    gripper_position: int
    temperature: Dict[int, float]
    voltage: Dict[int, float]
    errors: List[str]
    timestamp: float
```

### AudioHandler

#### Class: `MacAudioHandler`

Handles voice recognition using OpenAI Whisper on macOS.

```python
class MacAudioHandler:
    def __init__(self, config: Optional[AudioConfig] = None)
```

**Methods:**

##### `initialize() -> bool`
Initialize audio system and load Whisper model.

##### `listen_once() -> Optional[Dict[str, Any]]`
Listen for a single voice command.

**Returns:** Dictionary with `text`, `confidence`, and `language`

##### `listen_continuously() -> AsyncGenerator[Dict[str, Any], None]`
Continuously listen for voice commands.

##### `test_audio_input(duration: float = 5.0) -> bool`
Test microphone and audio processing.

##### `get_audio_devices() -> Dict[str, List[Dict]]`
Get available audio input devices.

#### Class: `AudioConfig`

Configuration for audio processing.

```python
@dataclass
class AudioConfig:
    sample_rate: int = 16000
    chunk_size: int = 1024
    whisper_model: str = "base"         # tiny, base, small, medium, large
    vad_threshold: float = 0.5
    silence_timeout: float = 2.0
    phrase_timeout: float = 3.0
    device_index: Optional[int] = None
```

### AICommandProcessor

#### Class: `AICommandProcessor`

AI-powered natural language command interpretation.

```python
class AICommandProcessor:
    def __init__(self, config: Optional[AIConfig] = None)
```

**Methods:**

##### `initialize() -> bool`
Initialize AI models and processing pipeline.

##### `process_command_async(command: str, context: Optional[Dict] = None) -> Dict[str, Any]`
Process natural language command using AI.

**Parameters:**
- `command`: Natural language command text
- `context`: Optional context information

**Returns:** Dictionary with `intent`, `action`, `response`, `confidence`

##### `add_context(context_type: str, data: Any) -> None`
Add context information for better command understanding.

#### Class: `AIConfig`

Configuration for AI processing.

```python
@dataclass
class AIConfig:
    primary_model: str = "Qwen/Qwen2.5-3B-Instruct"
    fallback_model: str = "microsoft/Phi-3.5-mini-instruct"
    max_length: int = 512
    temperature: float = 0.7
    confidence_threshold: float = 0.7
    enable_context: bool = True
    device: str = "auto"                # auto, mps, cuda, cpu
```

### CameraHandler

#### Class: `MacCameraHandler`

Camera integration for computer vision on macOS.

```python
class MacCameraHandler:
    def __init__(self, config: Optional[CameraConfig] = None)
```

**Methods:**

##### `start_capture(device_id: Optional[str] = None) -> bool`
Start camera capture.

**Parameters:**
- `device_id`: Specific camera device ID (optional)

##### `stop_capture() -> None`
Stop camera capture.

##### `get_frame() -> Optional[np.ndarray]`
Get latest captured frame.

**Returns:** Numpy array representing the image frame

##### `get_available_cameras() -> List[CameraDevice]`
Get list of available camera devices.

##### `capture_single_frame(device_id: Optional[str] = None) -> Optional[np.ndarray]`
Capture a single frame without continuous capture.

##### `test_camera_access() -> bool`
Test camera permissions and functionality.

#### Class: `CameraConfig`

Configuration for camera capture.

```python
@dataclass
class CameraConfig:
    device_id: Optional[str] = None
    resolution: Tuple[int, int] = (1280, 720)
    fps: int = 30
    enable_continuity_camera: bool = True
    enable_usb_cameras: bool = True
    enable_builtin_camera: bool = True
    buffer_size: int = 5
```

### ObjectDetector

#### Class: `ObjectDetector`

Computer vision object detection and tracking.

```python
class ObjectDetector:
    def __init__(self, config: Optional[DetectionConfig] = None)
```

**Methods:**

##### `initialize() -> bool`
Initialize object detection models.

##### `detect_objects_async(frame: np.ndarray) -> List[Detection]`
Detect objects in image frame.

**Parameters:**
- `frame`: Input image as numpy array

**Returns:** List of `Detection` objects

##### `get_detection_stats() -> Dict[str, Any]`
Get performance statistics for object detection.

##### `visualize_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray`
Render detection overlays on image frame.

#### Class: `Detection`

Single object detection result.

```python
class Detection(NamedTuple):
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]     # x, y, width, height
    center: Tuple[int, int]             # center x, y
    area: float
```

#### Class: `DetectionConfig`

Configuration for object detection.

```python
@dataclass
class DetectionConfig:
    model_name: str = "facebook/detr-resnet-50"
    confidence_threshold: float = 0.7
    nms_threshold: float = 0.5
    max_detections: int = 100
    input_size: Tuple[int, int] = (800, 600)
    enable_tracking: bool = True
    device: str = "auto"
```

### ContextManager

#### Class: `ContextManager`

Manages conversation and spatial context for AI processing.

```python
class ContextManager:
    def __init__(self)
```

**Methods:**

##### `add_conversation_turn(speaker: str, text: str, intent: Optional[str] = None, entities: Optional[List[str]] = None) -> None`
Add conversation turn to context history.

**Parameters:**
- `speaker`: "user" or "assistant"
- `text`: Conversation text
- `intent`: Optional detected intent
- `entities`: Optional list of mentioned entities

##### `track_object(object_name: str, properties: Optional[Dict[str, Any]] = None, position: Optional[Tuple[float, float, float]] = None) -> None`
Track object in spatial context.

##### `resolve_command_references(command_text: str) -> str`
Resolve pronouns and references in command text.

##### `get_spatial_reference(reference: str) -> Optional[Tuple[float, float, float]]`
Get 3D coordinates for spatial reference.

##### `update_visual_detections(detections: List[Detection], frame: Optional[np.ndarray] = None) -> None`
Update context with visual detection information.

##### `get_context_summary() -> Dict[str, Any]`
Get comprehensive summary of current context.

### MultimodalInterface

#### Class: `MultimodalInterface`

Integrated voice and vision interface.

```python
class MultimodalInterface:
    def __init__(self, config: Optional[MultimodalConfig] = None)
```

**Methods:**

##### `initialize() -> bool`
Initialize all multimodal components.

##### `start() -> None`
Start multimodal processing loop.

##### `stop() -> None`
Stop multimodal interface.

##### `get_current_visual_frame() -> Optional[np.ndarray]`
Get current camera frame with overlays.

##### `get_interface_stats() -> Dict[str, Any]`
Get performance statistics for multimodal interface.

#### Class: `MultimodalConfig`

Configuration for multimodal interface.

```python
@dataclass
class MultimodalConfig:
    enable_voice: bool = True
    enable_vision: bool = True
    enable_visual_feedback: bool = True
    camera_resolution: Tuple[int, int] = (1280, 720)
    detection_confidence: float = 0.6
    spatial_reference_resolution: bool = True
    max_processing_fps: int = 30
```

## Utility Modules

### SafetyMonitor

#### Class: `SafetyMonitor`

Monitors robot safety and workspace limits.

```python
class SafetyMonitor:
    def __init__(self, config: SafetyConfig)
```

**Methods:**

##### `start_monitoring() -> None`
Start safety monitoring.

##### `stop_monitoring() -> None`
Stop safety monitoring.

##### `check_workspace_limits(robot_id: str, joint_positions: Dict[int, float]) -> bool`
Verify joint positions are within safe workspace limits.

##### `update_motor_status(motor_id: int, position: float, velocity: float, current: float, temperature: float, voltage: float, has_error: bool) -> None`
Update motor status for safety monitoring.

##### `get_safety_status() -> Dict[str, Any]`
Get current safety status information.

#### Class: `SafetyConfig`

Safety monitoring configuration.

```python
@dataclass
class SafetyConfig:
    workspace_limits: Dict[int, Tuple[float, float]]
    max_velocity: float = 100.0
    max_temperature: float = 70.0
    min_voltage: float = 11.0
    max_current: float = 2.0
    emergency_stop_enabled: bool = True
```

### PerformanceMonitor

#### Class: `PerformanceMonitor`

System performance monitoring and metrics.

```python
class PerformanceMonitor:
    def __init__(self, config: Optional[PerformanceConfig] = None)
```

**Methods:**

##### `start_monitoring() -> None`
Start performance monitoring.

##### `stop_monitoring() -> None`
Stop performance monitoring.

##### `start_timing(operation: str) -> None`
Start timing an operation.

##### `end_timing(operation: str) -> float`
End timing and return duration.

##### `get_performance_metrics() -> Dict[str, Any]`
Get comprehensive performance metrics.

## Configuration Classes

### System Configuration

#### Class: `SystemConfig`

Overall system configuration.

```python
@dataclass
class SystemConfig:
    audio: AudioConfig
    robot: RobotConfig
    ai: AIConfig
    vision: CameraConfig
    safety: SafetyConfig
    performance: PerformanceConfig
```

## Exception Classes

### Custom Exceptions

#### `SaorsaError`
Base exception class for Saorsa-specific errors.

#### `RobotConnectionError`
Raised when robot connection fails.

#### `SafetyViolationError`
Raised when safety limits are exceeded.

#### `AIProcessingError`
Raised when AI processing fails.

#### `VisionError`
Raised when computer vision processing fails.

## Constants and Enumerations

### Robot Constants

```python
# Joint limits (degrees)
JOINT_LIMITS = {
    1: (-150, 150),
    2: (-90, 90),
    3: (-90, 90),
    4: (-180, 180),
    5: (-90, 90),
    6: (-180, 180)
}

# Default speeds
DEFAULT_SPEED = 50
MAX_SPEED = 100
MIN_SPEED = 1

# Gripper limits
GRIPPER_OPEN = 100
GRIPPER_CLOSED = 0
```

### AI Model Constants

```python
# Supported models
LANGUAGE_MODELS = [
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct"
]

VISION_MODELS = [
    "facebook/detr-resnet-50",
    "facebook/detr-resnet-101",
    "yolov8n",
    "yolov8s"
]
```

## Usage Examples

### Basic Robot Control

```python
from src.robot_controller_m3 import RobotController, create_default_so101_config

# Create robot configuration
config = create_default_so101_config("/dev/tty.usbserial-FT1234", "Leader")

# Initialize robot controller
robot = RobotController(config)

# Connect to robot
if robot.connect("/dev/tty.usbserial-FT1234"):
    # Move to home position
    robot.home_position()
    
    # Move specific joint
    robot.move_joint("leader", 1, 45.0, speed=30)
    
    # Control gripper
    robot.set_gripper("leader", 50)  # Half open
    
    # Disconnect
    robot.disconnect()
```

### Voice Recognition

```python
from src.mac_audio_handler import MacAudioHandler, AudioConfig

# Create audio configuration
config = AudioConfig(whisper_model="base", vad_threshold=0.5)

# Initialize audio handler
audio = MacAudioHandler(config)

# Test audio input
if audio.initialize() and audio.test_audio_input():
    # Listen for command
    result = audio.listen_once()
    if result:
        print(f"Command: {result['text']}")
        print(f"Confidence: {result['confidence']}")
```

### AI Command Processing

```python
from src.ai_command_processor import AICommandProcessor
from src.context_manager import ContextManager

# Initialize components
ai_processor = AICommandProcessor()
context_manager = ContextManager()

# Initialize AI processor
if ai_processor.initialize():
    # Process command
    command = "pick up the red block"
    context_manager.add_conversation_turn("user", command)
    
    response = ai_processor.process_command_async(command)
    print(f"Intent: {response['intent']}")
    print(f"Action: {response['action']}")
```

### Computer Vision

```python
from src.mac_camera_handler import MacCameraHandler
from src.object_detector import ObjectDetector
from src.visual_feedback import VisualFeedback

# Initialize components
camera = MacCameraHandler()
detector = ObjectDetector()
feedback = VisualFeedback()

# Start camera and detection
if camera.start_capture() and detector.initialize():
    # Process frames
    frame = camera.get_frame()
    if frame is not None:
        # Detect objects
        detections = detector.detect_objects_async(frame)
        
        # Update visual feedback
        feedback.update_detections(detections)
        annotated_frame = feedback.render_frame(frame)
        
        # Display results
        print(f"Found {len(detections)} objects")
        for det in detections:
            print(f"  {det.class_name}: {det.confidence:.2f}")
```

### Multimodal Integration

```python
from src.multimodal_interface import MultimodalInterface, MultimodalConfig

# Create configuration
config = MultimodalConfig(
    enable_voice=True,
    enable_vision=True,
    camera_resolution=(1280, 720)
)

# Initialize interface
interface = MultimodalInterface(config)

# Start multimodal processing
if interface.initialize():
    # This starts voice + vision processing
    interface.start()
```

## Error Handling

### Common Error Patterns

```python
from src.robot_controller_m3 import RobotController, RobotConnectionError
from src.exceptions import SafetyViolationError

try:
    robot = RobotController(config)
    robot.connect(port)
    robot.move_joint("leader", 1, 180)  # May exceed limits
    
except RobotConnectionError as e:
    print(f"Connection failed: {e}")
    
except SafetyViolationError as e:
    print(f"Safety violation: {e}")
    robot.emergency_stop()
    
except Exception as e:
    print(f"Unexpected error: {e}")
    
finally:
    if robot:
        robot.disconnect()
```

This API reference provides comprehensive documentation for integrating with and extending the Saorsa robot control system.