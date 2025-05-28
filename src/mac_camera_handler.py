#!/usr/bin/env python3
"""
macOS Camera Handler for Saorsa Robot System

This module provides camera integration using macOS AVFoundation framework
with support for built-in cameras, USB cameras, and iPhone Continuity Camera.
"""

import asyncio
import logging
import threading
import time
from typing import Optional, Callable, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import queue
import sys

import numpy as np
import cv2
from rich.console import Console

try:
    # macOS specific imports
    import objc
    from Foundation import NSObject, NSNotificationCenter, NSRunLoop, NSDefaultRunLoopMode
    from AVFoundation import (
        AVCaptureSession, AVCaptureDevice, AVCaptureDeviceInput,
        AVCaptureVideoDataOutput, AVCaptureDeviceTypeBuiltInWideAngleCamera,
        AVCaptureDeviceTypeExternal, AVCaptureDeviceTypeContinuityCamera,
        AVMediaTypeVideo, AVCaptureSessionPresetHigh, AVCaptureSessionPreset640x480,
        AVCaptureSessionPreset1280x720, AVCaptureSessionPreset1920x1080,
        kCVPixelFormatType_32BGRA
    )
    from CoreVideo import CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow
    from Quartz import CGImageRef
    
    MACOS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"macOS AVFoundation not available: {e}")
    MACOS_AVAILABLE = False

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration settings."""
    device_id: Optional[str] = None
    resolution: Tuple[int, int] = (1280, 720)
    fps: int = 30
    enable_continuity_camera: bool = True
    enable_usb_cameras: bool = True
    enable_builtin_camera: bool = True
    auto_exposure: bool = True
    auto_focus: bool = True
    buffer_size: int = 5


@dataclass
class CameraDevice:
    """Information about an available camera device."""
    device_id: str
    name: str
    device_type: str
    is_connected: bool = True
    supports_continuity: bool = False
    resolution: Optional[Tuple[int, int]] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)


class AVCaptureVideoDataOutputSampleBufferDelegate(NSObject):
    """Delegate to handle video frame capture from AVFoundation."""
    
    def initWithHandler_(self, handler):
        """Initialize with frame handler callback."""
        self = objc.super(AVCaptureVideoDataOutputSampleBufferDelegate, self).init()
        if self is None:
            return None
        self.frame_handler = handler
        return self
        
    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample_buffer, connection):
        """Called when a new video frame is available."""
        try:
            if self.frame_handler:
                self.frame_handler(sample_buffer)
        except Exception as e:
            logger.error(f"Frame processing error: {e}")


class MacCameraHandler:
    """
    macOS camera handler using AVFoundation.
    
    Provides camera capture with support for built-in cameras, USB cameras,
    and iPhone Continuity Camera with optimizations for Apple Silicon.
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        if not MACOS_AVAILABLE:
            raise RuntimeError("macOS AVFoundation not available on this system")
            
        self.config = config or CameraConfig()
        
        # AVFoundation components
        self.capture_session = None
        self.camera_device = None
        self.device_input = None
        self.video_output = None
        self.delegate = None
        
        # Frame management
        self.frame_queue = queue.Queue(maxsize=self.config.buffer_size)
        self.latest_frame = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # State management
        self.is_capturing = False
        self.capture_thread = None
        
        # Callbacks
        self.frame_callbacks: List[Callable[[np.ndarray], None]] = []
        
    def get_available_cameras(self) -> List[CameraDevice]:
        """Get list of available camera devices."""
        devices = []
        
        try:
            # Get all video devices
            device_types = []
            
            # Built-in cameras
            if self.config.enable_builtin_camera:
                device_types.append(AVCaptureDeviceTypeBuiltInWideAngleCamera)
                
            # USB/External cameras
            if self.config.enable_usb_cameras:
                device_types.append(AVCaptureDeviceTypeExternal)
                
            # Continuity Camera (iPhone/iPad)
            if self.config.enable_continuity_camera:
                device_types.append(AVCaptureDeviceTypeContinuityCamera)
                
            # Discovery session for multiple device types
            from AVFoundation import AVCaptureDeviceDiscoverySession
            
            discovery_session = AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
                device_types, AVMediaTypeVideo, None
            )
            
            for device in discovery_session.devices():
                device_info = CameraDevice(
                    device_id=device.uniqueID(),
                    name=device.localizedName(),
                    device_type=self._get_device_type_name(device.deviceType()),
                    is_connected=device.isConnected(),
                    supports_continuity=device.deviceType() == AVCaptureDeviceTypeContinuityCamera
                )
                
                # Get supported resolutions
                device_info.resolution = self._get_best_resolution(device)
                device_info.capabilities = self._get_device_capabilities(device)
                
                devices.append(device_info)
                
        except Exception as e:
            logger.error(f"Error discovering cameras: {e}")
            
        return devices
        
    def _get_device_type_name(self, device_type) -> str:
        """Convert device type to readable string."""
        type_mapping = {
            AVCaptureDeviceTypeBuiltInWideAngleCamera: "Built-in Camera",
            AVCaptureDeviceTypeExternal: "USB/External Camera",
            AVCaptureDeviceTypeContinuityCamera: "iPhone/iPad (Continuity Camera)"
        }
        return type_mapping.get(device_type, "Unknown Camera")
        
    def _get_best_resolution(self, device) -> Tuple[int, int]:
        """Get the best supported resolution for a device."""
        try:
            # Try common resolutions in order of preference
            preferred_resolutions = [
                (1920, 1080),
                (1280, 720),
                (640, 480)
            ]
            
            for width, height in preferred_resolutions:
                # This is a simplified check - real implementation would
                # query actual supported formats
                return (width, height)
                
        except Exception:
            return (640, 480)  # Fallback
            
    def _get_device_capabilities(self, device) -> Dict[str, Any]:
        """Get device capabilities."""
        capabilities = {}
        
        try:
            capabilities["has_auto_focus"] = device.isFocusModeSupported_(1)  # Auto focus
            capabilities["has_auto_exposure"] = device.isExposureModeSupported_(2)  # Auto exposure
            capabilities["has_torch"] = device.hasTorch()
            capabilities["supports_tap_to_focus"] = device.isFocusPointOfInterestSupported()
            
        except Exception as e:
            logger.warning(f"Could not get device capabilities: {e}")
            
        return capabilities
        
    def start_capture(self, device_id: Optional[str] = None) -> bool:
        """Start camera capture."""
        
        if self.is_capturing:
            logger.warning("Camera capture already started")
            return True
            
        try:
            # Create capture session
            self.capture_session = AVCaptureSession.alloc().init()
            self.capture_session.beginConfiguration()
            
            # Find and configure camera device
            if not self._setup_camera_device(device_id):
                return False
                
            # Setup video output
            if not self._setup_video_output():
                return False
                
            # Set session preset (resolution)
            preset = self._get_session_preset()
            if self.capture_session.canSetSessionPreset_(preset):
                self.capture_session.setSessionPreset_(preset)
                
            self.capture_session.commitConfiguration()
            
            # Start capture session
            self.capture_session.startRunning()
            
            self.is_capturing = True
            console.print(f"[green]✓ Camera capture started[/green]")
            
            # Start processing thread
            self.capture_thread = threading.Thread(
                target=self._capture_loop, 
                daemon=True
            )
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera capture: {e}")
            console.print(f"[red]✗ Camera capture failed: {e}[/red]")
            self.stop_capture()
            return False
            
    def _setup_camera_device(self, device_id: Optional[str]) -> bool:
        """Setup camera device input."""
        
        try:
            # Get camera device
            if device_id:
                self.camera_device = AVCaptureDevice.deviceWithUniqueID_(device_id)
            else:
                # Use default device
                self.camera_device = AVCaptureDevice.defaultDeviceWithMediaType_(AVMediaTypeVideo)
                
            if not self.camera_device:
                logger.error("No camera device found")
                return False
                
            # Create device input
            error = None
            self.device_input = AVCaptureDeviceInput.deviceInputWithDevice_error_(
                self.camera_device, objc.nil
            )
            
            if not self.device_input:
                logger.error(f"Failed to create device input: {error}")
                return False
                
            # Add input to session
            if self.capture_session.canAddInput_(self.device_input):
                self.capture_session.addInput_(self.device_input)
            else:
                logger.error("Cannot add camera input to session")
                return False
                
            # Configure camera settings
            self._configure_camera_settings()
            
            return True
            
        except Exception as e:
            logger.error(f"Camera device setup failed: {e}")
            return False
            
    def _configure_camera_settings(self):
        """Configure camera settings like focus and exposure."""
        
        try:
            if self.camera_device.lockForConfiguration_(objc.nil):
                
                # Auto focus
                if (self.config.auto_focus and 
                    self.camera_device.isFocusModeSupported_(1)):  # AVCaptureFocusModeAutoFocus
                    self.camera_device.setFocusMode_(1)
                    
                # Auto exposure
                if (self.config.auto_exposure and
                    self.camera_device.isExposureModeSupported_(2)):  # AVCaptureExposureModeAutoExpose
                    self.camera_device.setExposureMode_(2)
                    
                self.camera_device.unlockForConfiguration()
                
        except Exception as e:
            logger.warning(f"Could not configure camera settings: {e}")
            
    def _setup_video_output(self) -> bool:
        """Setup video data output."""
        
        try:
            # Create video output
            self.video_output = AVCaptureVideoDataOutput.alloc().init()
            
            # Set pixel format (BGRA for easy conversion to RGB)
            video_settings = {
                'kCVPixelBufferPixelFormatTypeKey': kCVPixelFormatType_32BGRA
            }
            self.video_output.setVideoSettings_(video_settings)
            
            # Set delegate for frame processing
            self.delegate = AVCaptureVideoDataOutputSampleBufferDelegate.alloc().initWithHandler_(
                self._process_sample_buffer
            )
            
            # Create dispatch queue for video processing
            from dispatch import dispatch_queue_create, DISPATCH_QUEUE_SERIAL
            video_queue = dispatch_queue_create(b"video_queue", DISPATCH_QUEUE_SERIAL)
            self.video_output.setSampleBufferDelegate_queue_(self.delegate, video_queue)
            
            # Don't drop frames for real-time processing
            self.video_output.setAlwaysDiscardsLateVideoFrames_(True)
            
            # Add output to session
            if self.capture_session.canAddOutput_(self.video_output):
                self.capture_session.addOutput_(self.video_output)
                return True
            else:
                logger.error("Cannot add video output to session")
                return False
                
        except Exception as e:
            logger.error(f"Video output setup failed: {e}")
            return False
            
    def _get_session_preset(self) -> str:
        """Get appropriate session preset for configured resolution."""
        
        width, height = self.config.resolution
        
        if width >= 1920 and height >= 1080:
            return AVCaptureSessionPreset1920x1080
        elif width >= 1280 and height >= 720:
            return AVCaptureSessionPreset1280x720
        elif width >= 640 and height >= 480:
            return AVCaptureSessionPreset640x480
        else:
            return AVCaptureSessionPresetHigh
            
    def _process_sample_buffer(self, sample_buffer):
        """Process video sample buffer from AVFoundation."""
        
        try:
            from CoreVideo import (
                CVPixelBufferLockBaseAddress, CVPixelBufferUnlockBaseAddress,
                CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
                CVPixelBufferGetWidth, CVPixelBufferGetHeight,
                kCVPixelBufferLock_ReadOnly
            )
            from AVFoundation import CMSampleBufferGetImageBuffer
            
            # Get pixel buffer from sample buffer
            pixel_buffer = CMSampleBufferGetImageBuffer(sample_buffer)
            
            if pixel_buffer:
                # Lock pixel buffer
                CVPixelBufferLockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly)
                
                try:
                    # Get buffer properties
                    base_address = CVPixelBufferGetBaseAddress(pixel_buffer)
                    bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer)
                    width = CVPixelBufferGetWidth(pixel_buffer)
                    height = CVPixelBufferGetHeight(pixel_buffer)
                    
                    # Convert to numpy array
                    buffer_length = bytes_per_row * height
                    frame_data = objc.PyObjC_PythonFromCFData(base_address, buffer_length)
                    
                    # Convert BGRA to RGB
                    frame = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = frame.reshape((height, width, 4))  # BGRA
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    
                    # Update frame queue
                    try:
                        if not self.frame_queue.full():
                            self.frame_queue.put_nowait(frame)
                        else:
                            # Remove oldest frame and add new one
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                            self.frame_queue.put_nowait(frame)
                            
                        self.latest_frame = frame
                        self.frame_count += 1
                        
                    except Exception as queue_error:
                        logger.warning(f"Frame queue error: {queue_error}")
                        
                finally:
                    # Always unlock the pixel buffer
                    CVPixelBufferUnlockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly)
                    
        except Exception as e:
            logger.error(f"Sample buffer processing error: {e}")
            
    def _capture_loop(self):
        """Main capture loop for processing frames."""
        
        while self.is_capturing:
            try:
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps_counter = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = current_time
                    
                # Process frame callbacks
                if self.latest_frame is not None:
                    for callback in self.frame_callbacks:
                        try:
                            callback(self.latest_frame.copy())
                        except Exception as e:
                            logger.error(f"Frame callback error: {e}")
                            
                time.sleep(0.01)  # Small sleep to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(0.1)
                
    def stop_capture(self):
        """Stop camera capture."""
        
        self.is_capturing = False
        
        try:
            if self.capture_session and self.capture_session.isRunning():
                self.capture_session.stopRunning()
                
            # Clean up inputs and outputs
            if self.capture_session:
                if self.device_input:
                    self.capture_session.removeInput_(self.device_input)
                if self.video_output:
                    self.capture_session.removeOutput_(self.video_output)
                    
            # Reset components
            self.capture_session = None
            self.camera_device = None
            self.device_input = None
            self.video_output = None
            self.delegate = None
            
            # Wait for capture thread
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1.0)
                
            console.print("[yellow]Camera capture stopped[/yellow]")
            
        except Exception as e:
            logger.error(f"Error stopping capture: {e}")
            
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame."""
        return self.latest_frame.copy() if self.latest_frame is not None else None
        
    def get_frame_async(self) -> Optional[np.ndarray]:
        """Get frame from queue (non-blocking)."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
            
    def add_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Add callback for processing frames."""
        self.frame_callbacks.append(callback)
        
    def remove_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Remove frame callback."""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
            
    def get_capture_info(self) -> Dict[str, Any]:
        """Get information about current capture session."""
        
        info = {
            "is_capturing": self.is_capturing,
            "fps": self.fps_counter,
            "frame_count": self.frame_count,
            "queue_size": self.frame_queue.qsize(),
            "device_name": None,
            "resolution": self.config.resolution
        }
        
        if self.camera_device:
            info["device_name"] = self.camera_device.localizedName()
            info["device_type"] = self._get_device_type_name(self.camera_device.deviceType())
            info["device_id"] = self.camera_device.uniqueID()
            
        return info
        
    def capture_single_frame(self, device_id: Optional[str] = None) -> Optional[np.ndarray]:
        """Capture a single frame without starting continuous capture."""
        
        if self.is_capturing:
            return self.get_frame()
            
        # Temporarily start capture for single frame
        if self.start_capture(device_id):
            # Wait for first frame
            for _ in range(50):  # Max 5 seconds wait
                time.sleep(0.1)
                frame = self.get_frame()
                if frame is not None:
                    self.stop_capture()
                    return frame
                    
            self.stop_capture()
            
        return None
        
    def test_camera_access(self) -> bool:
        """Test camera access and permissions."""
        
        try:
            from AVFoundation import AVCaptureDevice, AVAuthorizationStatusAuthorized
            
            # Check camera permissions
            auth_status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
            
            if auth_status != AVAuthorizationStatusAuthorized:
                console.print("[yellow]Camera permission not granted[/yellow]")
                console.print("[yellow]Please allow camera access in System Preferences > Privacy & Security > Camera[/yellow]")
                return False
                
            # Try to get default camera
            device = AVCaptureDevice.defaultDeviceWithMediaType_(AVMediaTypeVideo)
            if not device:
                console.print("[red]No camera device found[/red]")
                return False
                
            console.print(f"[green]✓ Camera access OK: {device.localizedName()}[/green]")
            return True
            
        except Exception as e:
            logger.error(f"Camera access test failed: {e}")
            console.print(f"[red]✗ Camera access test failed: {e}[/red]")
            return False


async def main():
    """Example usage of MacCameraHandler."""
    
    if not MACOS_AVAILABLE:
        console.print("[red]macOS AVFoundation not available[/red]")
        return
        
    console.print("[blue]Saorsa Camera Handler Demo[/blue]\n")
    
    # Test camera access
    handler = MacCameraHandler()
    
    if not handler.test_camera_access():
        return
        
    # List available cameras
    cameras = handler.get_available_cameras()
    
    if not cameras:
        console.print("[red]No cameras found[/red]")
        return
        
    console.print("[blue]Available cameras:[/blue]")
    for i, camera in enumerate(cameras):
        console.print(f"  {i}: {camera.name} ({camera.device_type})")
        if camera.supports_continuity:
            console.print(f"      [green]✓ Supports Continuity Camera[/green]")
            
    # Test single frame capture
    console.print("\n[blue]Testing single frame capture...[/blue]")
    frame = handler.capture_single_frame()
    
    if frame is not None:
        console.print(f"[green]✓ Captured frame: {frame.shape}[/green]")
        
        # Save test frame
        import cv2
        cv2.imwrite("/tmp/saorsa_test_frame.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        console.print("[green]✓ Test frame saved to /tmp/saorsa_test_frame.jpg[/green]")
    else:
        console.print("[red]✗ Failed to capture frame[/red]")
        
    # Test continuous capture for a few seconds
    console.print("\n[blue]Testing continuous capture for 3 seconds...[/blue]")
    
    frame_count = 0
    def count_frames(frame):
        nonlocal frame_count
        frame_count += 1
        
    handler.add_frame_callback(count_frames)
    
    if handler.start_capture():
        await asyncio.sleep(3)
        
        info = handler.get_capture_info()
        console.print(f"[green]✓ Captured {frame_count} frames at ~{info['fps']} FPS[/green]")
        
        handler.stop_capture()
    else:
        console.print("[red]✗ Failed to start continuous capture[/red]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())