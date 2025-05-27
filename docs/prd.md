# Product Requirements Document (PRD)
## Saorse: Voice-Controlled SO-101 Robot Arms System

### Executive Summary

Saorse is a voice-controlled robotic system that enables natural language control of Hugging Face's SO-101 robot arms using a Mac M3 computer. The system leverages OpenAI's Whisper for speech recognition and Physical Intelligence's π0 (pi-zero) foundation model for advanced command interpretation, providing an intuitive interface for robotic manipulation tasks.

### Project Overview

**Project Name:** Saorse  
**Repository:** https://github.com/dirvine/saorse  
**Target Platform:** macOS (Apple Silicon M3)  
**Primary Language:** Python 3.11+  
**License:** Apache 2.0

### Objectives

1. Create a natural language interface for controlling SO-101 robot arms
2. Leverage Mac M3's performance capabilities for local AI processing
3. Provide a modular, extensible architecture for custom commands
4. Ensure safety and reliability in robot control
5. Enable both simple commands and complex AI-guided actions

### Technical Requirements

#### System Requirements
- macOS with Apple Silicon (M3, M3 Pro, or M3 Max)
- Python 3.11 or higher
- Homebrew package manager
- Minimum 16GB RAM (32GB recommended)
- 20GB free disk space for models

#### Hardware Requirements
- SO-101 Robot Arms (2x - leader and follower configuration)
- USB-to-Serial adapters (2x) for robot communication
- Built-in or external microphone
- Optional: Camera for visual commands (built-in or iPhone via Continuity Camera)

#### Software Dependencies
- PyTorch with Metal Performance Shaders (MPS) support
- OpenAI Whisper for speech recognition
- LeRobot framework from Hugging Face
- Dynamixel SDK for motor control
- Core audio/video frameworks for macOS

### Architecture

#### System Components

1. **Voice Input Module** (`mac_audio_handler.py`)
   - Continuous audio capture using macOS Core Audio
   - Real-time speech-to-text using Whisper
   - Voice activity detection for efficiency
   - Support for multiple Whisper model sizes

2. **Robot Controller** (`robot_controller_m3.py`)
   - Serial communication with SO-101 arms
   - Motor control via Dynamixel protocol
   - Safety checks and workspace limits
   - Emergency stop functionality

3. **AI Command Processor**
   - Integration with π0 foundation model
   - Natural language understanding
   - Context-aware command interpretation
   - Complex task planning

4. **Camera Module** (`mac_camera_handler.py`)
   - macOS AVFoundation integration
   - iPhone Continuity Camera support
   - Real-time frame capture for visual commands

5. **Main Application** (`main_mac.py`)
   - Command routing and orchestration
   - Asynchronous processing pipeline
   - User interface and feedback

### Core Features

#### Voice Commands

**Basic Commands:**
- Movement: "move left", "move right", "move forward", "move back"
- Gripper: "open gripper", "close gripper"
- Positions: "home position", "ready position"
- Actions: "pick up", "put down", "grab", "release"
- Control: "stop", "emergency stop", "slower", "faster"

**Advanced Commands (AI-powered):**
- Object-specific: "pick up the red block"
- Contextual: "put it over there"
- Sequential: "stack the blocks by size"
- Descriptive: "arrange the items neatly"

#### Safety Features
- Workspace boundary enforcement
- Velocity limiting
- Emergency stop on command
- Collision detection (future)
- Automatic error recovery

#### Performance Optimizations
- MPS acceleration for PyTorch models
- Efficient audio processing pipeline
- Model quantization options
- Asynchronous command processing

### File Structure

```
saorse/
├── README.md                     # Project overview and quick start
├── LICENSE                       # Apache 2.0 license
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation script
├── launch.sh                     # Main launch script
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── main_mac.py              # Main application entry point
│   ├── mac_audio_handler.py     # Voice input processing
│   ├── robot_controller_m3.py   # Robot control logic
│   ├── mac_camera_handler.py    # Camera integration
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── performance_monitor.py
│       └── safety_monitor.py
│
├── configs/                      # Configuration files
│   ├── default.yaml             # Default settings
│   ├── mac_m3.yaml              # M3-specific optimizations
│   └── voice_commands.yaml      # Command mappings
│
├── models/                       # AI models directory
│   ├── README.md                # Model installation guide
│   └── .gitkeep
│
├── scripts/                      # Setup and utility scripts
│   ├── install_mac.sh           # One-click installation
│   ├── test_voice.py            # Voice system test
│   ├── calibrate_robot.py       # Robot calibration
│   └── download_models.py       # Model downloader
│
├── tests/                        # Test suite
│   ├── test_voice_controller.py
│   ├── test_robot_controller.py
│   └── test_integration.py
│
├── docs/                         # Documentation
│   ├── hardware_setup.md        # Hardware assembly guide
│   ├── software_setup.md        # Software installation
│   ├── voice_commands.md        # Command reference
│   └── troubleshooting.md       # Common issues
│
├── examples/                     # Example scripts
│   ├── basic_voice_control.py
│   ├── advanced_commands.py
│   └── custom_actions.py
│
└── .github/                      # GitHub configuration
    └── workflows/
        ├── ci.yml               # Continuous integration
        └── tests.yml            # Automated testing

```

### Implementation Phases

#### Phase 1: Core Infrastructure (Week 1)
1. Set up repository structure
2. Implement basic voice recognition
3. Establish robot communication
4. Create safety framework
5. Basic movement commands

#### Phase 2: AI Integration (Week 2)
1. Integrate Whisper models
2. Add π0 model support
3. Implement command interpretation
4. Context-aware processing
5. Advanced movement patterns

#### Phase 3: Visual Integration (Week 3)
1. Camera module implementation
2. Continuity Camera support
3. Basic object detection
4. Visual feedback system
5. Multi-modal commands

#### Phase 4: Polish & Documentation (Week 4)
1. Performance optimization
2. Comprehensive testing
3. Documentation completion
4. Example scripts
5. Video demonstrations

### API Design

#### Voice Controller API
```python
class VoiceController:
    def start_listening(callback: Callable) -> None
    def stop_listening() -> None
    def set_wake_word(word: str) -> None
    def set_language(lang: str) -> None
```

#### Robot Controller API
```python
class RobotController:
    def connect(leader_port: str, follower_port: str) -> bool
    def move_to_position(positions: Dict[str, float]) -> None
    def set_gripper(value: int) -> None
    def emergency_stop() -> None
    def get_current_state() -> Dict
```

#### Command Processor API
```python
class CommandProcessor:
    def process_command(text: str, context: Dict) -> Action
    def register_custom_command(pattern: str, handler: Callable) -> None
    def get_command_history() -> List[Command]
```

### Configuration

#### Default Configuration (configs/default.yaml)
```yaml
system:
  device: "mps"
  log_level: "INFO"

audio:
  sample_rate: 16000
  chunk_duration: 0.5
  wake_word: "robot"

whisper:
  model_size: "base"
  language: "en"

robot:
  baudrate: 1000000
  workspace_limits:
    x: [-300, 300]
    y: [-300, 300]
    z: [0, 400]

safety:
  max_velocity: 30
  emergency_words: ["stop", "halt", "emergency"]
```

### Testing Strategy

1. **Unit Tests**
   - Voice recognition accuracy
   - Command parsing correctness
   - Motor control precision
   - Safety boundary enforcement

2. **Integration Tests**
   - End-to-end command execution
   - Multi-modal command processing
   - Error recovery scenarios
   - Performance benchmarks

3. **System Tests**
   - Real robot testing
   - Voice command reliability
   - Response time measurement
   - Safety feature validation

### Success Metrics

1. **Performance Metrics**
   - Voice recognition accuracy > 95%
   - Command execution latency < 500ms
   - System uptime > 99%
   - Model inference time < 100ms

2. **User Experience Metrics**
   - Natural language understanding rate > 90%
   - Successful task completion > 85%
   - Error recovery success > 95%
   - User satisfaction score > 4.5/5

### Risk Mitigation

1. **Technical Risks**
   - Serial communication failures: Implement retry logic
   - Voice recognition errors: Provide visual feedback
   - Model loading failures: Include fallback commands
   - Hardware damage: Enforce strict safety limits

2. **Operational Risks**
   - Power loss: Graceful shutdown procedures
   - Network issues: Local-first architecture
   - Environmental noise: Advanced noise cancellation

### Future Enhancements

1. **Short-term (3 months)**
   - Multi-robot coordination
   - Custom voice training
   - Web-based control panel
   - Cloud model updates

2. **Long-term (6-12 months)**
   - Computer vision integration
   - Learning from demonstration
   - Natural gesture control
   - Multi-language support

### Development Guidelines

1. **Code Style**
   - Follow PEP 8 for Python code
   - Use type hints throughout
   - Comprehensive docstrings
   - Meaningful variable names

2. **Git Workflow**
   - Feature branches for new development
   - PR reviews required
   - Semantic versioning
   - Comprehensive commit messages

3. **Documentation**
   - Code comments for complex logic
   - API documentation
   - User guides with examples
   - Video tutorials

### Dependencies and Licensing

All dependencies must be compatible with Apache 2.0 license:
- PyTorch: BSD-style license ✓
- OpenAI Whisper: MIT license ✓
- LeRobot: Apache 2.0 ✓
- Dynamixel SDK: Apache 2.0 ✓

### Deliverables

1. **Week 1**
   - Repository with basic structure
   - Voice recognition prototype
   - Basic robot control

2. **Week 2**
   - AI model integration
   - Advanced commands
   - Initial documentation

3. **Week 3**
   - Camera integration
   - Visual commands
   - Test suite

4. **Week 4**
   - Complete documentation
   - Installation scripts
   - Demo videos
   - Public release

### Conclusion

Saorse represents a significant step forward in making robotic control accessible through natural language. By leveraging the power of Mac M3 hardware and state-of-the-art AI models, we create an intuitive, powerful, and safe system for controlling robotic arms. The modular architecture ensures extensibility while the focus on user experience makes advanced robotics accessible to a broader audience.
