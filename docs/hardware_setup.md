# Hardware Setup Guide

This guide covers the physical hardware setup for the Saorse voice-controlled SO-101 robot arms system.

## Required Hardware

### Computer Requirements

#### Minimum Requirements
- **Mac with Apple Silicon**: M1, M2, or M3 processor
- **RAM**: 16GB (32GB recommended for advanced AI features)
- **Storage**: 25GB free space for models and data
- **macOS**: 13.0 (Ventura) or later
- **USB Ports**: 1-2 USB-A or USB-C ports for robot connections

#### Recommended Specifications
- **Mac Studio M3 Max** or **MacBook Pro M3 Pro/Max**
- **32GB+ RAM** for optimal AI model performance
- **1TB+ SSD** with fast read/write speeds
- **Multiple USB ports** for robot arms and peripherals

### Robot Hardware

#### SO-101 Robot Arms
- **Primary Robot**: SO-101 6-DOF robot arm
- **Secondary Robot** (optional): Second SO-101 for dual-arm operations
- **Gripper**: Standard SO-101 gripper or compatible end effector
- **Power Supply**: 12V DC power adapter (included with robot)
- **Communication**: USB-to-serial adapter (usually included)

#### Robot Specifications
- **Degrees of Freedom**: 6 (6 servo motors)
- **Reach**: Approximately 350mm
- **Payload**: Up to 500g
- **Repeatability**: ±1mm
- **Communication**: UART over USB
- **Power**: 12V DC, 2A

### Audio Hardware

#### Built-in Options
- **MacBook/iMac**: Built-in microphone (adequate for basic use)
- **Mac Studio/Mac Pro**: External microphone required

#### Recommended External Microphones
- **Blue Yeti**: Professional USB microphone with multiple pickup patterns
- **Audio-Technica ATR2100x-USB**: Dynamic microphone with excellent noise rejection
- **Rode PodMic**: Broadcast-quality dynamic microphone
- **Shure SM7B**: Professional broadcasting microphone (requires audio interface)

#### Microphone Placement
- **Distance**: 12-18 inches from speaker
- **Angle**: Pointed toward speaker's mouth
- **Environment**: Reduce background noise and echo
- **Isolation**: Use pop filter and shock mount if available

### Camera Hardware (Phase 3)

#### Built-in Cameras
- **MacBook**: Built-in FaceTime HD camera
- **iMac**: Built-in FaceTime HD camera
- **Studio Display**: Built-in 12MP Ultra Wide camera

#### External Camera Options
- **Logitech C920/C922**: Popular USB webcams with good quality
- **Razer Kiyo**: Streaming camera with built-in ring light
- **Sony FX30**: Professional camera with HDMI capture
- **DSLR/Mirrorless**: Via capture card for highest quality

#### iPhone Continuity Camera
- **Requirements**: iPhone 8 or later with iOS 16+
- **Connection**: Wireless via Continuity Camera feature
- **Quality**: Often better than built-in Mac cameras
- **Setup**: Automatically detected when iPhone is nearby

#### Camera Positioning
- **Height**: Eye level with robot workspace
- **Distance**: 2-4 feet from robot for optimal object detection
- **Angle**: Slight downward angle to capture workspace
- **Lighting**: Ensure good lighting on robot workspace
- **Background**: Minimize clutter behind robot

## Physical Setup

### Workspace Layout

#### Robot Positioning
```
[Camera]
    |
    v
[Workspace Area]
    |
[Robot Base] ---- [Computer]
    |
[Power Supply]
```

#### Recommended Workspace
- **Size**: Minimum 2x2 feet clear area
- **Surface**: Flat, stable table or workbench
- **Height**: Comfortable for viewing and manual intervention
- **Lighting**: Bright, even lighting for camera
- **Safety**: Clear of obstacles and breakable items

### Robot Assembly

#### SO-101 Assembly Steps
1. **Unbox** robot arm and verify all components
2. **Attach base** to stable mounting surface
3. **Connect joints** following manufacturer's assembly guide
4. **Install gripper** or end effector
5. **Route cables** neatly to avoid interference
6. **Connect power** and USB communication cables

#### Cable Management
- **Power cables**: Route away from moving parts
- **Communication cables**: Use strain relief and cable ties
- **Camera cables**: Secure to prevent movement during operation
- **Audio cables**: Keep away from motors to prevent electrical interference

### Safety Considerations

#### Workspace Safety
- **Emergency stop**: Physical emergency stop button accessible
- **Clear area**: No people or obstacles in robot reach
- **Secure mounting**: Robot base firmly attached to table
- **Cable routing**: All cables secured and out of motion paths
- **Lighting**: Adequate lighting for safe operation

#### Electrical Safety
- **Grounding**: Ensure all equipment is properly grounded
- **Power strips**: Use surge-protected power strips
- **Voltage**: Verify correct voltage for robot power supply
- **Inspection**: Regular inspection of cables for wear

## Connection Setup

### Robot Communication

#### USB-to-Serial Setup
1. **Connect** USB cable from robot to Mac
2. **Verify** connection: `ls /dev/tty.usbserial-*`
3. **Note** device name (e.g., `/dev/tty.usbserial-FT1234`)
4. **Test** communication with robot test script

#### Multiple Robot Setup
```bash
# Leader robot (primary)
/dev/tty.usbserial-FT1234

# Follower robot (secondary)  
/dev/tty.usbserial-FT5678
```

#### Communication Parameters
- **Baud Rate**: 1000000 (1M baud)
- **Data Bits**: 8
- **Stop Bits**: 1
- **Parity**: None
- **Flow Control**: None

### Audio Setup

#### macOS Audio Permissions
1. **System Preferences** → **Security & Privacy** → **Privacy**
2. **Microphone** → Enable access for Terminal and Python
3. **Test** microphone access in System Preferences

#### Audio Device Selection
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone
python src/main_mac.py test-audio
```

### Camera Setup (Phase 3)

#### macOS Camera Permissions
1. **System Preferences** → **Security & Privacy** → **Privacy**
2. **Camera** → Enable access for Terminal and Python
3. **Test** camera access

#### iPhone Continuity Camera Setup
1. **iPhone**: iOS 16+ with same Apple ID
2. **Mac**: macOS Ventura+ with same Apple ID
3. **Proximity**: iPhone within Bluetooth/WiFi range
4. **Automatic**: Should appear as camera option

## Testing Hardware Setup

### Initial Hardware Tests

#### Test Robot Connection
```bash
# Test single robot
python src/main_mac.py test-robot --port /dev/tty.usbserial-FT1234

# Test dual robots
python src/main_mac.py test-robot --port /dev/tty.usbserial-FT1234 --follower /dev/tty.usbserial-FT5678
```

#### Test Audio System
```bash
# Test microphone and speech recognition
python src/main_mac.py test-audio
```

#### Test Camera System
```bash
# Test camera access and detection
python src/main_mac.py test-camera
```

### System Status Check
```bash
# Comprehensive system status
python src/main_mac.py status --verbose
```

### Performance Validation

#### Robot Performance
- **Movement**: Smooth, accurate joint movements
- **Speed**: Appropriate movement speed (not too fast/slow)
- **Position**: Accurate positioning within tolerance
- **Gripper**: Proper opening/closing action

#### Audio Performance
- **Recognition**: Voice commands recognized accurately
- **Latency**: Minimal delay between speech and recognition
- **Noise**: Background noise rejection working
- **Volume**: Appropriate microphone sensitivity

#### Camera Performance (Phase 3)
- **Image Quality**: Clear, well-lit images
- **Frame Rate**: Smooth video capture (30 FPS target)
- **Detection**: Objects properly detected and identified
- **Tracking**: Consistent object tracking across frames

## Troubleshooting Hardware Issues

### Robot Connection Problems

#### No Serial Device Found
```bash
# Check USB connections
ls /dev/tty.usbserial-*

# Check System Information
System Information → Hardware → USB
```

#### Robot Not Responding
- **Power**: Verify robot power LED is on
- **Cables**: Check all cable connections
- **Baud Rate**: Try different baud rates
- **Reset**: Power cycle robot and restart software

### Audio Issues

#### Microphone Not Detected
- **Permissions**: Check macOS privacy settings
- **Connection**: Verify USB microphone connection
- **Drivers**: Update audio drivers if needed
- **Test**: Use built-in Sound preferences to test

#### Poor Voice Recognition
- **Distance**: Adjust microphone distance (12-18 inches)
- **Noise**: Reduce background noise
- **Environment**: Improve room acoustics
- **Calibration**: Run audio calibration script

### Camera Issues (Phase 3)

#### Camera Not Detected
- **Permissions**: Check macOS privacy settings for camera
- **Connection**: Verify USB camera connection
- **Drivers**: Check for camera driver updates
- **Test**: Use Photo Booth to verify camera function

#### Poor Object Detection
- **Lighting**: Improve workspace lighting
- **Background**: Reduce background clutter
- **Distance**: Adjust camera distance to workspace
- **Angle**: Optimize camera angle for workspace view

## Maintenance

### Regular Maintenance

#### Daily Checks
- **Power connections**: Verify all power connections secure
- **Movement**: Test basic robot movements
- **Audio**: Quick voice recognition test
- **Camera**: Verify camera feed quality

#### Weekly Maintenance
- **Cable inspection**: Check all cables for wear
- **Cleaning**: Clean robot joints and gripper
- **Camera lens**: Clean camera lens for optimal image quality
- **Software updates**: Check for system updates

#### Monthly Maintenance
- **Calibration**: Re-calibrate robot positioning if needed
- **Performance**: Run full system performance tests
- **Backup**: Backup configuration and calibration data
- **Documentation**: Update any configuration changes

### Hardware Upgrades

#### Robot Upgrades
- **End Effectors**: Different grippers or tools
- **Sensors**: Additional sensors for feedback
- **Mounting**: Improved mounting solutions
- **Workspace**: Larger or specialized work surfaces

#### Computer Upgrades
- **RAM**: Additional memory for larger AI models
- **Storage**: More storage for model cache and logs
- **GPU**: External GPU for enhanced AI performance (future)
- **Networking**: Faster networking for remote operation

This hardware setup guide provides the foundation for a reliable, high-performance Saorse robot control system. Proper hardware setup is crucial for optimal system performance and safety.