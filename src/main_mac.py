#!/usr/bin/env python3
"""
Main Application Entry Point for Saorse Robot Control System

This module provides the main application orchestration, command routing,
and user interface for voice-controlled SO-101 robot arms on Mac M3.
"""

import asyncio
import logging
import signal
import sys
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import yaml
import click

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text

# Import Saorse modules
from mac_audio_handler import MacAudioHandler, AudioConfig
from robot_controller_m3 import RobotController, create_default_so101_config, RobotState
from utils.safety_monitor import SafetyMonitor, create_default_safety_config, SafetyAlert
from utils.performance_monitor import PerformanceMonitor

# Phase 2 & 3 imports (AI and Vision)
try:
    from ai_command_processor import AICommandProcessor
    from context_manager import ContextManager
    from multimodal_interface import MultimodalInterface, MultimodalConfig
    from mac_camera_handler import MacCameraHandler
    from object_detector import ObjectDetector
    from visual_feedback import VisualFeedback
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ADVANCED_FEATURES_AVAILABLE = False
    logger.warning(f"Advanced features not available: {e}")

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class CommandContext:
    """Context information for command processing."""
    text: str
    timestamp: float
    confidence: float = 1.0
    wake_word_detected: bool = False
    processed: bool = False


class CommandProcessor:
    """
    Command processor for natural language robot control.
    
    Handles command interpretation, validation, and execution routing.
    """
    
    def __init__(self, robot_controller: RobotController, safety_monitor: SafetyMonitor):
        self.robot = robot_controller
        self.safety = safety_monitor
        
        # Command mappings
        self.basic_commands = {
            # Movement commands
            "home": ("move_home", {}),
            "home position": ("move_home", {}),
            "ready": ("move_ready", {}),
            "ready position": ("move_ready", {}),
            
            # Gripper commands
            "open gripper": ("set_gripper", {"value": 100}),
            "close gripper": ("set_gripper", {"value": 0}),
            "open": ("set_gripper", {"value": 100}),
            "close": ("set_gripper", {"value": 0}),
            "grab": ("set_gripper", {"value": 0}),
            "release": ("set_gripper", {"value": 100}),
            
            # Control commands
            "stop": ("stop_movement", {}),
            "halt": ("emergency_stop", {}),
            "emergency": ("emergency_stop", {}),
            "emergency stop": ("emergency_stop", {}),
            "freeze": ("emergency_stop", {}),
        }
        
        # Direction mappings
        self.movement_commands = {
            "move left": ("move_relative", {"joint1": -10}),
            "move right": ("move_relative", {"joint1": 10}),
            "move forward": ("move_relative", {"joint2": -10}),
            "move back": ("move_relative", {"joint2": 10}),
            "move backward": ("move_relative", {"joint2": 10}),
            "move up": ("move_relative", {"joint3": 10}),
            "move down": ("move_relative", {"joint3": -10}),
            "turn left": ("move_relative", {"joint6": -15}),
            "turn right": ("move_relative", {"joint6": 15}),
        }
        
        # Speed modifiers
        self.speed_modifiers = {
            "slower": 0.5,
            "faster": 2.0,
            "slow": 0.3,
            "fast": 3.0,
        }
        
        self.current_speed = 1.0
        
    def process_command(self, text: str, context: Dict[str, Any] = None) -> bool:
        """Process voice command and execute appropriate action."""
        context = context or {}
        text_lower = text.lower().strip()
        
        # Check for safety issues first
        safety_alert = self.safety.process_voice_command(text)
        if safety_alert:
            return True  # Safety system will handle emergency stop
            
        # Check for speed modifiers
        for modifier, factor in self.speed_modifiers.items():
            if modifier in text_lower:
                self.current_speed = factor
                console.print(f"[blue]Speed set to {factor}x")
                return True
                
        # Check basic commands
        for command_phrase, (action, params) in self.basic_commands.items():
            if command_phrase in text_lower:
                return self._execute_action(action, params)
                
        # Check movement commands
        for movement_phrase, (action, params) in self.movement_commands.items():
            if movement_phrase in text_lower:
                # Apply speed modifier
                if "joint" in str(params):
                    for joint, value in params.items():
                        params[joint] = value * self.current_speed
                return self._execute_action(action, params)
                
        # Check for position commands (numbers)
        if self._contains_position_numbers(text_lower):
            return self._process_position_command(text_lower)
            
        # If no command matched, log it
        console.print(f"[yellow]❓ Unknown command: '{text}'")
        logger.info(f"Unrecognized command: {text}")
        return False
        
    def _execute_action(self, action: str, params: Dict[str, Any]) -> bool:
        """Execute a specific robot action."""
        try:
            if action == "move_home":
                self.robot.home_position()
                console.print("[green]🏠 Moving to home position")
                
            elif action == "move_ready":
                self.robot.ready_position()
                console.print("[green]📍 Moving to ready position")
                
            elif action == "set_gripper":
                value = params.get("value", 50)
                self.robot.set_gripper(value)
                state = "open" if value > 50 else "closed"
                console.print(f"[green]🤏 Gripper {state}")
                
            elif action == "move_relative":
                # Convert relative movement to absolute positions
                current_state = self.robot.get_current_state()
                # This is simplified - real implementation would use forward kinematics
                self.robot.move_to_position(params)
                joint_str = ", ".join([f"{k}: {v}°" for k, v in params.items()])
                console.print(f"[green]➡️  Moving joints: {joint_str}")
                
            elif action == "stop_movement":
                # Note: This would need to be implemented in robot controller
                console.print("[yellow]⏹️  Stop command received")
                
            elif action == "emergency_stop":
                self.robot.emergency_stop()
                console.print("[red bold]🚨 EMERGENCY STOP")
                
            else:
                logger.warning(f"Unknown action: {action}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            console.print(f"[red]❌ Command failed: {e}")
            return False
            
    def _contains_position_numbers(self, text: str) -> bool:
        """Check if text contains position/angle numbers."""
        # Simple check for degree values
        import re
        degree_pattern = r'\d+\s*(?:degree|deg|°)'
        return bool(re.search(degree_pattern, text))
        
    def _process_position_command(self, text: str) -> bool:
        """Process position-specific commands with numbers."""
        # This would be expanded to parse specific position commands
        # like "move joint 1 to 45 degrees"
        console.print("[yellow]📐 Position command parsing not fully implemented")
        return False


class SaorseApplication:
    """
    Main Saorse application class.
    
    Orchestrates all components and provides the main application loop.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/default.yaml"
        self.config = self._load_config()
        
        # Initialize components
        self.audio_handler: Optional[MacAudioHandler] = None
        self.robot_controller: Optional[RobotController] = None
        self.safety_monitor: Optional[SafetyMonitor] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.command_processor: Optional[CommandProcessor] = None
        
        # Advanced components (Phase 2 & 3)
        self.ai_processor: Optional['AICommandProcessor'] = None
        self.context_manager: Optional['ContextManager'] = None
        self.enable_ai_mode = False
        
        # Application state
        self.is_running = False
        self.is_connected = False
        self.shutdown_event = asyncio.Event()
        
        # Statistics
        self.start_time = time.time()
        self.commands_processed = 0
        self.commands_successful = 0
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
            
        # Return default configuration
        return {
            "system": {"device": "mps", "log_level": "INFO"},
            "audio": {
                "sample_rate": 16000,
                "chunk_duration": 0.5,
                "wake_word": "robot"
            },
            "whisper": {"model_size": "base", "language": "en"},
            "robot": {
                "baudrate": 1000000,
                "workspace_limits": {
                    "x": [-300, 300],
                    "y": [-300, 300],
                    "z": [0, 400]
                }
            },
            "safety": {
                "max_velocity": 30,
                "emergency_words": ["stop", "halt", "emergency"]
            }
        }
        
    async def initialize(self) -> bool:
        """Initialize all application components."""
        try:
            console.print("[yellow]🚀 Initializing Saorse...")
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor()
            self.performance_monitor.add_warning_callback(self._performance_warning)
            self.performance_monitor.start_monitoring()
            
            # Initialize safety monitor
            workspace_limits, safety_limits = create_default_safety_config()
            self.safety_monitor = SafetyMonitor(workspace_limits, safety_limits)
            self.safety_monitor.add_alert_callback(self._safety_alert)
            self.safety_monitor.set_emergency_stop_callback(self._emergency_stop)
            self.safety_monitor.start_monitoring()
            
            # Initialize audio handler
            audio_config = AudioConfig(
                sample_rate=self.config["audio"]["sample_rate"],
                chunk_duration=self.config["audio"]["chunk_duration"],
                wake_word=self.config["audio"]["wake_word"]
            )
            self.audio_handler = MacAudioHandler(audio_config)
            
            # Test audio
            console.print("[yellow]🎤 Testing audio system...")
            if not self.audio_handler.test_audio_input(duration=2.0):
                console.print("[red]❌ Audio test failed")
                return False
                
            console.print("[green]✓ Audio system ready")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            console.print(f"[red]❌ Initialization failed: {e}")
            return False
            
    async def connect_robot(self, leader_port: str, follower_port: Optional[str] = None) -> bool:
        """Connect to robot arms."""
        try:
            console.print(f"[yellow]🤖 Connecting to robot on {leader_port}...")
            
            # Create robot configurations
            leader_config = create_default_so101_config(leader_port, "Leader")
            follower_config = None
            if follower_port:
                follower_config = create_default_so101_config(follower_port, "Follower")
                
            # Initialize robot controller
            self.robot_controller = RobotController(leader_config, follower_config)
            self.robot_controller.set_status_callback(self._robot_status_update)
            
            # Connect
            if self.robot_controller.connect(leader_port, follower_port):
                self.is_connected = True
                
                # Initialize command processor
                self.command_processor = CommandProcessor(
                    self.robot_controller, self.safety_monitor
                )
                
                # Initialize AI components if enabled
                if self.enable_ai_mode and ADVANCED_FEATURES_AVAILABLE:
                    asyncio.create_task(self._initialize_ai_components())
                
                console.print("[green]✓ Robot connected successfully")
                return True
            else:
                console.print("[red]❌ Robot connection failed")
                return False
                
        except Exception as e:
            logger.error(f"Robot connection failed: {e}")
            console.print(f"[red]❌ Robot connection failed: {e}")
            return False
            
    async def start_voice_control(self) -> None:
        """Start voice control system."""
        if not self.audio_handler or not self.command_processor:
            console.print("[red]❌ Voice control not available")
            return
            
        console.print("[green]🎤 Starting voice control...")
        self.audio_handler.start_listening(self._voice_command_callback)
        
    def stop_voice_control(self) -> None:
        """Stop voice control system."""
        if self.audio_handler:
            self.audio_handler.stop_listening()
            
    async def run(self) -> None:
        """Main application run loop."""
        self.is_running = True
        
        try:
            # Display startup banner
            self._display_banner()
            
            # Start voice control if robot is connected
            if self.is_connected:
                await self.start_voice_control()
                
            # Main application loop
            while self.is_running and not self.shutdown_event.is_set():
                # Update display
                self._update_display()
                
                # Sleep briefly
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Received interrupt signal")
        except Exception as e:
            logger.error(f"Application error: {e}")
            console.print(f"[red]Application error: {e}")
        finally:
            await self.shutdown()
            
    async def shutdown(self) -> None:
        """Gracefully shutdown the application."""
        console.print("[yellow]🔄 Shutting down Saorse...")
        
        self.is_running = False
        
        # Stop voice control
        self.stop_voice_control()
        
        # Disconnect robot
        if self.robot_controller:
            self.robot_controller.disconnect()
            
        # Stop monitoring
        if self.safety_monitor:
            self.safety_monitor.stop_monitoring()
            
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
            
        console.print("[green]✓ Saorse shutdown complete")
        
    def _voice_command_callback(self, text: str) -> None:
        """Handle voice command from audio system."""
        self.commands_processed += 1
        
        # Record performance
        if self.performance_monitor:
            self.performance_monitor.start_timing("command_processing")
            
        # Process command
        success = False
        
        if self.enable_ai_mode and self.ai_processor and ADVANCED_FEATURES_AVAILABLE:
            # Use AI processing
            asyncio.create_task(self._process_ai_command(text))
            success = True  # Assume success for AI commands
        elif self.command_processor:
            # Use traditional command processing
            success = self.command_processor.process_command(text)
            
        if success:
            self.commands_successful += 1
            
        # Record performance
        if self.performance_monitor:
            processing_time = self.performance_monitor.end_timing("command_processing")
            # This would be part of robot metrics in a real implementation
            
    async def _process_ai_command(self, text: str):
        """Process command using AI system."""
        try:
            if not self.ai_processor:
                return
                
            # Add to context if available
            if self.context_manager:
                self.context_manager.add_conversation_turn("user", text)
                
            # Process with AI
            response = await self.ai_processor.process_command_async(text)
            
            # Show AI response
            console.print(f"[cyan]AI: {response.get('response', 'Processing...')}[/cyan]")
            
            # Execute action if robot controller is available
            if response.get('action') and self.command_processor:
                action_success = self.command_processor.process_command(response['action'])
                if action_success:
                    console.print("[green]✓ Action executed[/green]")
                else:
                    console.print("[yellow]⚠ Action execution failed[/yellow]")
                    
            # Add AI response to context
            if self.context_manager:
                self.context_manager.add_conversation_turn(
                    "assistant", 
                    response.get('response', ''),
                    intent=response.get('intent')
                )
                
        except Exception as e:
            logger.error(f"AI command processing error: {e}")
            console.print(f"[red]AI processing error: {e}[/red]")
            
    async def _initialize_ai_components(self):
        """Initialize AI components for enhanced processing."""
        try:
            console.print("[blue]🧠 Initializing AI components...[/blue]")
            
            # Initialize context manager
            if ADVANCED_FEATURES_AVAILABLE:
                from context_manager import ContextManager
                from ai_command_processor import AICommandProcessor
                
                self.context_manager = ContextManager()
                console.print("[green]✓ Context manager initialized[/green]")
                
                # Initialize AI processor
                self.ai_processor = AICommandProcessor()
                if await self.ai_processor.initialize():
                    console.print("[green]✓ AI processor initialized[/green]")
                    console.print("[cyan]🧠 AI mode enabled - Enhanced command processing active[/cyan]")
                else:
                    console.print("[yellow]⚠ AI processor initialization failed[/yellow]")
                    self.enable_ai_mode = False
                    
        except Exception as e:
            logger.error(f"AI initialization error: {e}")
            console.print(f"[red]AI initialization failed: {e}[/red]")
            self.enable_ai_mode = False
            
    def _robot_status_update(self, status) -> None:
        """Handle robot status updates."""
        # Update safety monitor with motor statuses
        if self.safety_monitor and hasattr(status, 'motors'):
            for motor_status in status.motors:
                self.safety_monitor.update_motor_status(
                    motor_status.id,
                    motor_status.position,
                    motor_status.velocity,
                    motor_status.current,
                    motor_status.temperature,
                    motor_status.voltage,
                    motor_status.has_error
                )
                
    def _safety_alert(self, alert: SafetyAlert) -> None:
        """Handle safety alerts."""
        level_colors = {
            "safe": "green",
            "warning": "yellow", 
            "critical": "red",
            "emergency": "red bold"
        }
        color = level_colors.get(alert.level.value, "white")
        console.print(f"[{color}]🚨 SAFETY {alert.level.value.upper()}: {alert.message}")
        
    def _emergency_stop(self) -> None:
        """Handle emergency stop."""
        if self.robot_controller:
            self.robot_controller.emergency_stop()
            
    def _performance_warning(self, category: str, message: str) -> None:
        """Handle performance warnings."""
        console.print(f"[orange1]⚠️  PERFORMANCE {category}: {message}")
        
    def _display_banner(self) -> None:
        """Display startup banner."""
        banner = Panel.fit(
            "[bold blue]Saorse[/bold blue]\n"
            "[dim]Voice-Controlled SO-101 Robot Arms[/dim]\n"
            f"[dim]Status: {'Connected' if self.is_connected else 'Not Connected'}[/dim]",
            title="🤖 Robot Control System",
            border_style="blue"
        )
        console.print(banner)
        
    def _update_display(self) -> None:
        """Update application display (simplified for now)."""
        # In a full implementation, this would update a rich live display
        # with current status, metrics, etc.
        pass
        
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            console.print(f"\n[yellow]Received signal {signum}")
            self.shutdown_event.set()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


# CLI Interface
@click.group()
@click.option('--config', '-c', default='configs/default.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Saorse: Voice-Controlled SO-101 Robot Arms System"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@cli.command()
@click.option('--leader-port', '-l', required=True, help='Leader robot serial port')
@click.option('--follower-port', '-f', help='Follower robot serial port (optional)')
@click.option('--mode', '-m', type=click.Choice(['basic', 'ai', 'multimodal']), default='basic', 
              help='Operation mode: basic (voice only), ai (AI-enhanced), multimodal (voice+vision)')
@click.pass_context
async def run(ctx, leader_port, follower_port, mode):
    """Run the main Saorse application."""
    
    if mode in ['ai', 'multimodal'] and not ADVANCED_FEATURES_AVAILABLE:
        console.print("[red]❌ Advanced features not available. Please install required dependencies.")
        console.print("[yellow]Run: pip install -r requirements.txt")
        return
        
    if mode == 'multimodal':
        # Run multimodal interface
        await run_multimodal_interface(ctx, leader_port, follower_port)
    else:
        # Run traditional interface
        app = SaorseApplication(ctx.obj['config'])
        app._setup_signal_handlers()
        
        # Initialize
        if not await app.initialize():
            console.print("[red]❌ Initialization failed")
            return
            
        # Connect robot
        if not await app.connect_robot(leader_port, follower_port):
            console.print("[red]❌ Robot connection failed")
            return
            
        # Set AI mode if requested
        if mode == 'ai' and ADVANCED_FEATURES_AVAILABLE:
            app.enable_ai_mode = True
            
        # Run application
        await app.run()


@cli.command()
@click.pass_context
async def test_audio(ctx):
    """Test audio input system."""
    app = SaorseApplication(ctx.obj['config'])
    
    if not await app.initialize():
        return
        
    console.print("[green]✓ Audio test completed")


@cli.command()
@click.option('--port', '-p', required=True, help='Robot serial port')
@click.pass_context
async def test_robot(ctx, port):
    """Test robot connection."""
    app = SaorseApplication(ctx.obj['config'])
    
    if not await app.initialize():
        return
        
    if await app.connect_robot(port):
        console.print("[green]✓ Robot test completed")
        app.robot_controller.home_position()
        await asyncio.sleep(2)
        app.robot_controller.disconnect()
    else:
        console.print("[red]❌ Robot test failed")


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and configuration."""
    app = SaorseApplication(ctx.obj['config'])
    
    table = Table(title="Saorse System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    # Check audio devices
    try:
        from mac_audio_handler import MacAudioHandler
        handler = MacAudioHandler()
        devices = handler.get_audio_devices()
        table.add_row("Audio Input", f"✓ {len(devices['input_devices'])} devices")
    except Exception as e:
        table.add_row("Audio Input", f"❌ {e}")
        
    # Check PyTorch/MPS
    try:
        import torch
        mps_available = torch.backends.mps.is_available()
        table.add_row("PyTorch MPS", "✓ Available" if mps_available else "❌ Not available")
    except Exception:
        table.add_row("PyTorch MPS", "❌ PyTorch not installed")
        
    console.print(table)


async def run_multimodal_interface(ctx, leader_port: str, follower_port: Optional[str]):
    """Run the multimodal interface with vision and AI."""
    
    if not ADVANCED_FEATURES_AVAILABLE:
        console.print("[red]❌ Multimodal features not available")
        return
        
    console.print("[blue]🤖👁️🎤 Starting Saorse Multimodal Interface[/blue]")
    
    # Create multimodal configuration
    config = MultimodalConfig(
        enable_voice=True,
        enable_vision=True,
        enable_visual_feedback=True,
        detection_confidence=0.6,
        camera_resolution=(1280, 720),
        spatial_reference_resolution=True
    )
    
    # Initialize multimodal interface
    interface = MultimodalInterface(config)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        console.print("\n[yellow]Shutting down multimodal interface...[/yellow]")
        asyncio.create_task(interface.stop())
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start interface
        await interface.start()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Multimodal interface error: {e}[/red]")
        logger.error(f"Multimodal interface error: {e}")
    finally:
        await interface.stop()


@cli.command()
@click.pass_context
async def demo_vision(ctx):
    """Demo computer vision capabilities."""
    if not ADVANCED_FEATURES_AVAILABLE:
        console.print("[red]❌ Vision features not available")
        return
        
    console.print("[blue]👁️ Starting Vision Demo[/blue]")
    
    from mac_camera_handler import MacCameraHandler, CameraConfig
    from object_detector import ObjectDetector, DetectionConfig
    from visual_feedback import VisualFeedback, OverlayConfig
    
    # Initialize camera
    camera_config = CameraConfig(resolution=(1280, 720))
    camera = MacCameraHandler(camera_config)
    
    # Initialize object detector
    detection_config = DetectionConfig(confidence_threshold=0.5)
    detector = ObjectDetector(detection_config)
    
    # Initialize visual feedback
    overlay_config = OverlayConfig(show_detections=True, fps_display=True)
    feedback = VisualFeedback(overlay_config)
    
    try:
        # Start camera
        if not camera.start_capture():
            console.print("[red]❌ Camera initialization failed")
            return
            
        # Initialize detector
        if not await detector.initialize():
            console.print("[red]❌ Object detector initialization failed")
            return
            
        console.print("[green]✓ Vision demo running for 30 seconds...")
        console.print("[dim]Press Ctrl+C to stop early[/dim]")
        
        # Demo loop
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 30:
            frame = camera.get_frame()
            if frame is not None:
                # Run detection
                detections = await detector.detect_objects_async(frame)
                
                # Update feedback
                feedback.update_detections(detections)
                
                # Show results every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    console.print(f"[dim]Detected {len(detections)} objects: {[d.class_name for d in detections]}[/dim]")
                    
            await asyncio.sleep(0.1)
            
        # Show final stats
        stats = detector.get_detection_stats()
        console.print(f"[green]✓ Demo completed: {stats['fps']:.1f} FPS, {frame_count} frames processed[/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Vision demo error: {e}[/red]")
    finally:
        camera.stop_capture()
        detector.cleanup()


@cli.command()
@click.pass_context  
async def demo_ai(ctx):
    """Demo AI command processing capabilities."""
    if not ADVANCED_FEATURES_AVAILABLE:
        console.print("[red]❌ AI features not available")
        return
        
    console.print("[blue]🧠 Starting AI Demo[/blue]")
    
    from ai_command_processor import AICommandProcessor
    from context_manager import ContextManager
    
    # Initialize components
    ai_processor = AICommandProcessor()
    context_manager = ContextManager()
    
    try:
        # Initialize AI processor
        if not await ai_processor.initialize():
            console.print("[red]❌ AI processor initialization failed")
            return
            
        console.print("[green]✓ AI processor ready[/green]")
        
        # Demo commands
        demo_commands = [
            "pick up the red block",
            "move it to the left side", 
            "grab the blue cup",
            "place it on the table",
            "stack the blocks",
            "organize the objects by color"
        ]
        
        for command in demo_commands:
            console.print(f"\n[cyan]Processing: '{command}'[/cyan]")
            
            # Add to context
            context_manager.add_conversation_turn("user", command)
            
            # Process with AI
            response = await ai_processor.process_command_async(command)
            
            # Show response
            console.print(f"[green]Intent: {response.get('intent', 'unknown')}[/green]")
            console.print(f"[blue]Response: {response.get('response', 'No response')}[/blue]")
            
            if response.get('action'):
                console.print(f"[yellow]Action: {response['action']}[/yellow]")
                
            await asyncio.sleep(1)
            
        # Show context summary
        summary = context_manager.get_context_summary()
        console.print(f"\n[blue]Context Summary:[/blue]")
        console.print(f"Commands processed: {summary['session_info']['total_commands']}")
        console.print(f"Conversation turns: {summary['conversation']['turns']}")
        
    except Exception as e:
        console.print(f"[red]❌ AI demo error: {e}[/red]")


@cli.command()
@click.pass_context
async def test_camera(ctx):
    """Test camera system."""
    if not ADVANCED_FEATURES_AVAILABLE:
        console.print("[red]❌ Camera features not available")
        return
        
    from mac_camera_handler import MacCameraHandler
    
    handler = MacCameraHandler()
    
    # Test camera access
    if not handler.test_camera_access():
        return
        
    # List cameras
    cameras = handler.get_available_cameras()
    console.print(f"[green]✓ Found {len(cameras)} cameras[/green]")
    
    for camera in cameras:
        console.print(f"  📷 {camera.name} ({camera.device_type})")
        
    # Test single frame capture
    console.print("[blue]Testing frame capture...[/blue]")
    frame = handler.capture_single_frame()
    
    if frame is not None:
        console.print(f"[green]✓ Captured frame: {frame.shape}[/green]")
    else:
        console.print("[red]❌ Frame capture failed[/red]")


def main():
    """Main entry point."""
    # Handle async CLI commands
    import inspect
    
    def async_command(f):
        def wrapper(*args, **kwargs):
            if inspect.iscoroutinefunction(f):
                return asyncio.run(f(*args, **kwargs))
            return f(*args, **kwargs)
        return wrapper
        
    # Apply async wrapper to commands
    for name, command in cli.commands.items():
        if hasattr(command, 'callback') and inspect.iscoroutinefunction(command.callback):
            command.callback = async_command(command.callback)
            
    cli()


if __name__ == "__main__":
    main()