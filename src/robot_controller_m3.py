#!/usr/bin/env python3
"""
Robot Controller for SO-101 Arms on Mac M3

This module provides low-level control interface for Hugging Face SO-101 robot arms
using the Dynamixel SDK, optimized for macOS and Apple Silicon performance.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

import serial
import numpy as np
from dynamixel_sdk import *
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


class MotorModel(Enum):
    """Supported Dynamixel motor models."""
    XM430_W350 = "XM430-W350"
    XM540_W270 = "XM540-W270"
    XL330_M288 = "XL330-M288"


@dataclass
class MotorConfig:
    """Configuration for individual motor."""
    motor_id: int
    model: MotorModel
    min_position: int = 0
    max_position: int = 4095
    max_velocity: int = 100
    max_current: int = 200
    position_offset: int = 0
    direction: int = 1  # 1 or -1


@dataclass
class RobotConfig:
    """Configuration for robot arm."""
    name: str
    port: str
    baudrate: int = 1000000
    protocol_version: float = 2.0
    motors: List[MotorConfig] = field(default_factory=list)
    workspace_limits: Dict[str, List[float]] = field(default_factory=lambda: {
        "x": [-300, 300],
        "y": [-300, 300], 
        "z": [0, 400]
    })


class RobotState(Enum):
    """Robot operational states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    MOVING = "moving"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class MotorStatus:
    """Current motor status."""
    id: int
    position: int
    velocity: int
    current: int
    temperature: int
    voltage: float
    is_moving: bool
    has_error: bool
    error_code: int = 0


@dataclass
class RobotStatus:
    """Complete robot status."""
    state: RobotState
    timestamp: float
    motors: List[MotorStatus]
    end_effector_position: Optional[Tuple[float, float, float]] = None
    gripper_position: Optional[int] = None
    last_error: Optional[str] = None


class DynamixelController:
    """Low-level Dynamixel motor controller."""
    
    # Control table addresses (Protocol 2.0)
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_GOAL_VELOCITY = 104
    ADDR_GOAL_CURRENT = 102
    ADDR_PRESENT_POSITION = 132
    ADDR_PRESENT_VELOCITY = 128
    ADDR_PRESENT_CURRENT = 126
    ADDR_PRESENT_TEMPERATURE = 146
    ADDR_PRESENT_VOLTAGE = 144
    ADDR_MOVING = 122
    ADDR_HARDWARE_ERROR_STATUS = 70
    
    def __init__(self, port: str, baudrate: int = 1000000, protocol_version: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        
        self.port_handler = None
        self.packet_handler = None
        self.group_sync_write_position = None
        self.group_sync_read_position = None
        self.is_connected = False
        
    def connect(self) -> bool:
        """Connect to Dynamixel motors."""
        try:
            # Initialize PortHandler and PacketHandler
            self.port_handler = PortHandler(self.port)
            self.packet_handler = PacketHandler(self.protocol_version)
            
            # Open port
            if not self.port_handler.openPort():
                logger.error(f"Failed to open port {self.port}")
                return False
                
            # Set baudrate
            if not self.port_handler.setBaudRate(self.baudrate):
                logger.error(f"Failed to set baudrate {self.baudrate}")
                return False
                
            # Initialize sync read/write
            self.group_sync_write_position = GroupSyncWrite(
                self.port_handler, self.packet_handler, 
                self.ADDR_GOAL_POSITION, 4
            )
            self.group_sync_read_position = GroupSyncRead(
                self.port_handler, self.packet_handler,
                self.ADDR_PRESENT_POSITION, 4
            )
            
            self.is_connected = True
            console.print(f"[green]✓ Connected to Dynamixel on {self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
            
    def disconnect(self) -> None:
        """Disconnect from motors."""
        if self.port_handler:
            self.port_handler.closePort()
        self.is_connected = False
        console.print("[yellow]Disconnected from Dynamixel")
        
    def ping_motor(self, motor_id: int) -> bool:
        """Ping individual motor to check connectivity."""
        if not self.is_connected:
            return False
            
        try:
            dxl_model_number, dxl_comm_result, dxl_error = self.packet_handler.ping(
                self.port_handler, motor_id
            )
            
            if dxl_comm_result != COMM_SUCCESS:
                logger.warning(f"Motor {motor_id} ping failed: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
                return False
            elif dxl_error != 0:
                logger.warning(f"Motor {motor_id} error: {self.packet_handler.getRxPacketError(dxl_error)}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Ping motor {motor_id} failed: {e}")
            return False
            
    def enable_torque(self, motor_id: int, enable: bool = True) -> bool:
        """Enable/disable motor torque."""
        if not self.is_connected:
            return False
            
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, self.ADDR_TORQUE_ENABLE, int(enable)
        )
        
        if dxl_comm_result != COMM_SUCCESS:
            logger.error(f"Failed to set torque for motor {motor_id}")
            return False
        elif dxl_error != 0:
            logger.error(f"Motor {motor_id} torque error: {self.packet_handler.getRxPacketError(dxl_error)}")
            return False
            
        return True
        
    def set_position(self, motor_id: int, position: int) -> bool:
        """Set motor goal position."""
        if not self.is_connected:
            return False
            
        dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
            self.port_handler, motor_id, self.ADDR_GOAL_POSITION, position
        )
        
        if dxl_comm_result != COMM_SUCCESS:
            logger.error(f"Failed to set position for motor {motor_id}")
            return False
        elif dxl_error != 0:
            logger.error(f"Motor {motor_id} position error: {self.packet_handler.getRxPacketError(dxl_error)}")
            return False
            
        return True
        
    def get_position(self, motor_id: int) -> Optional[int]:
        """Get current motor position."""
        if not self.is_connected:
            return None
            
        dxl_present_position, dxl_comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
            self.port_handler, motor_id, self.ADDR_PRESENT_POSITION
        )
        
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            return None
            
        return dxl_present_position
        
    def get_motor_status(self, motor_id: int) -> Optional[MotorStatus]:
        """Get complete motor status."""
        if not self.is_connected:
            return None
            
        try:
            # Read multiple values
            position, _, _ = self.packet_handler.read4ByteTxRx(
                self.port_handler, motor_id, self.ADDR_PRESENT_POSITION
            )
            velocity, _, _ = self.packet_handler.read4ByteTxRx(
                self.port_handler, motor_id, self.ADDR_PRESENT_VELOCITY
            )
            current, _, _ = self.packet_handler.read2ByteTxRx(
                self.port_handler, motor_id, self.ADDR_PRESENT_CURRENT
            )
            temperature, _, _ = self.packet_handler.read1ByteTxRx(
                self.port_handler, motor_id, self.ADDR_PRESENT_TEMPERATURE
            )
            voltage, _, _ = self.packet_handler.read2ByteTxRx(
                self.port_handler, motor_id, self.ADDR_PRESENT_VOLTAGE
            )
            moving, _, _ = self.packet_handler.read1ByteTxRx(
                self.port_handler, motor_id, self.ADDR_MOVING
            )
            error_status, _, _ = self.packet_handler.read1ByteTxRx(
                self.port_handler, motor_id, self.ADDR_HARDWARE_ERROR_STATUS
            )
            
            return MotorStatus(
                id=motor_id,
                position=position,
                velocity=velocity,
                current=current,
                temperature=temperature,
                voltage=voltage / 10.0,  # Convert to volts
                is_moving=bool(moving),
                has_error=bool(error_status),
                error_code=error_status
            )
            
        except Exception as e:
            logger.error(f"Failed to get status for motor {motor_id}: {e}")
            return None


class RobotController:
    """
    High-level robot controller for SO-101 arms.
    
    Provides safe, coordinated control of robot arm movements with
    workspace limits, safety monitoring, and emergency stop capability.
    """
    
    def __init__(self, leader_config: RobotConfig, follower_config: Optional[RobotConfig] = None):
        self.leader_config = leader_config
        self.follower_config = follower_config
        
        self.leader_controller = DynamixelController(
            leader_config.port, leader_config.baudrate, leader_config.protocol_version
        )
        self.follower_controller = None
        if follower_config:
            self.follower_controller = DynamixelController(
                follower_config.port, follower_config.baudrate, follower_config.protocol_version
            )
            
        self.state = RobotState.DISCONNECTED
        self.emergency_stop_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        
        # Current positions (motor positions)
        self.current_positions: Dict[str, Dict[int, int]] = {
            "leader": {},
            "follower": {}
        }
        
        # Safety limits
        self.workspace_limits = leader_config.workspace_limits
        self.max_velocity = 100  # Default safe velocity
        
        # Callbacks
        self.status_callback: Optional[Callable[[RobotStatus], None]] = None
        
    def connect(self, leader_port: str, follower_port: Optional[str] = None) -> bool:
        """Connect to robot arms."""
        self.state = RobotState.CONNECTING
        
        # Update port configurations
        self.leader_config.port = leader_port
        if follower_port and self.follower_config:
            self.follower_config.port = follower_port
            
        # Connect to leader
        if not self.leader_controller.connect():
            self.state = RobotState.ERROR
            return False
            
        # Connect to follower if configured
        if self.follower_controller:
            if not self.follower_controller.connect():
                logger.warning("Failed to connect to follower, continuing with leader only")
                self.follower_controller = None
                
        # Ping all motors
        if not self._ping_all_motors():
            self.state = RobotState.ERROR
            return False
            
        # Enable torque for all motors
        if not self._enable_all_torques():
            self.state = RobotState.ERROR
            return False
            
        self.state = RobotState.CONNECTED
        self._start_monitoring()
        
        console.print("[green]✓ Robot connected successfully")
        return True
        
    def disconnect(self) -> None:
        """Disconnect from robot arms."""
        self._stop_monitoring()
        
        # Disable torques before disconnecting
        self._disable_all_torques()
        
        self.leader_controller.disconnect()
        if self.follower_controller:
            self.follower_controller.disconnect()
            
        self.state = RobotState.DISCONNECTED
        
    def emergency_stop(self) -> None:
        """Immediate emergency stop of all motors."""
        self.emergency_stop_active = True
        self.state = RobotState.EMERGENCY_STOP
        
        # Disable all torques immediately
        self._disable_all_torques()
        
        console.print("[red bold]🚨 EMERGENCY STOP ACTIVATED 🚨")
        
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop and re-enable motors."""
        if not self.emergency_stop_active:
            return True
            
        # Re-enable torques
        if self._enable_all_torques():
            self.emergency_stop_active = False
            self.state = RobotState.CONNECTED
            console.print("[green]✓ Emergency stop reset")
            return True
        else:
            console.print("[red]✗ Failed to reset emergency stop")
            return False
            
    def move_to_position(self, positions: Dict[str, float], arm: str = "leader") -> None:
        """Move arm to specified joint positions (in degrees)."""
        if self.state != RobotState.CONNECTED or self.emergency_stop_active:
            logger.warning("Cannot move: robot not ready")
            return
            
        # Convert degrees to motor positions
        motor_positions = self._degrees_to_motor_positions(positions, arm)
        
        # Validate positions are within limits
        if not self._validate_positions(motor_positions, arm):
            logger.warning("Position validation failed")
            return
            
        self.state = RobotState.MOVING
        
        # Send positions to motors
        controller = self.leader_controller if arm == "leader" else self.follower_controller
        if not controller:
            logger.error(f"Controller for {arm} not available")
            return
            
        config = self.leader_config if arm == "leader" else self.follower_config
        
        for motor_config in config.motors:
            if motor_config.motor_id in motor_positions:
                position = motor_positions[motor_config.motor_id]
                controller.set_position(motor_config.motor_id, position)
                
        # Update current positions
        self.current_positions[arm].update(motor_positions)
        
        # Wait for movement completion (simplified)
        self._wait_for_movement_completion(arm)
        self.state = RobotState.CONNECTED
        
    def set_gripper(self, value: int, arm: str = "leader") -> None:
        """Set gripper position (0-100)."""
        if self.state != RobotState.CONNECTED or self.emergency_stop_active:
            return
            
        # Assuming gripper is the last motor in the configuration
        config = self.leader_config if arm == "leader" else self.follower_config
        if not config or not config.motors:
            return
            
        gripper_motor = config.motors[-1]  # Typically the last motor
        controller = self.leader_controller if arm == "leader" else self.follower_controller
        
        # Convert 0-100 to motor position range
        motor_position = int(
            gripper_motor.min_position + 
            (value / 100.0) * (gripper_motor.max_position - gripper_motor.min_position)
        )
        
        controller.set_position(gripper_motor.motor_id, motor_position)
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current robot state and positions."""
        state_data = {
            "state": self.state.value,
            "emergency_stop": self.emergency_stop_active,
            "timestamp": time.time(),
            "positions": self.current_positions.copy()
        }
        
        # Add motor statuses
        if self.state == RobotState.CONNECTED:
            state_data["motor_status"] = self._get_all_motor_status()
            
        return state_data
        
    def home_position(self, arm: str = "leader") -> None:
        """Move arm to home position."""
        # Define home positions for each joint (in degrees)
        home_positions = {
            "joint1": 0,
            "joint2": -45,
            "joint3": 45,
            "joint4": 0,
            "joint5": 0,
            "joint6": 0
        }
        
        self.move_to_position(home_positions, arm)
        console.print(f"[green]✓ {arm.title()} arm moved to home position")
        
    def ready_position(self, arm: str = "leader") -> None:
        """Move arm to ready position."""
        ready_positions = {
            "joint1": 0,
            "joint2": -30,
            "joint3": 60,
            "joint4": 0,
            "joint5": -30,
            "joint6": 0
        }
        
        self.move_to_position(ready_positions, arm)
        console.print(f"[green]✓ {arm.title()} arm moved to ready position")
        
    def set_status_callback(self, callback: Callable[[RobotStatus], None]) -> None:
        """Set callback for status updates."""
        self.status_callback = callback
        
    def _ping_all_motors(self) -> bool:
        """Ping all configured motors."""
        all_connected = True
        
        # Ping leader motors
        for motor in self.leader_config.motors:
            if not self.leader_controller.ping_motor(motor.motor_id):
                logger.error(f"Leader motor {motor.motor_id} not responding")
                all_connected = False
                
        # Ping follower motors if available
        if self.follower_controller and self.follower_config:
            for motor in self.follower_config.motors:
                if not self.follower_controller.ping_motor(motor.motor_id):
                    logger.error(f"Follower motor {motor.motor_id} not responding")
                    all_connected = False
                    
        return all_connected
        
    def _enable_all_torques(self) -> bool:
        """Enable torque for all motors."""
        success = True
        
        # Enable leader motors
        for motor in self.leader_config.motors:
            if not self.leader_controller.enable_torque(motor.motor_id, True):
                success = False
                
        # Enable follower motors if available
        if self.follower_controller and self.follower_config:
            for motor in self.follower_config.motors:
                if not self.follower_controller.enable_torque(motor.motor_id, True):
                    success = False
                    
        return success
        
    def _disable_all_torques(self) -> bool:
        """Disable torque for all motors."""
        success = True
        
        # Disable leader motors
        for motor in self.leader_config.motors:
            if not self.leader_controller.enable_torque(motor.motor_id, False):
                success = False
                
        # Disable follower motors if available
        if self.follower_controller and self.follower_config:
            for motor in self.follower_config.motors:
                if not self.follower_controller.enable_torque(motor.motor_id, False):
                    success = False
                    
        return success
        
    def _degrees_to_motor_positions(self, degrees: Dict[str, float], arm: str) -> Dict[int, int]:
        """Convert joint angles in degrees to motor positions."""
        config = self.leader_config if arm == "leader" else self.follower_config
        motor_positions = {}
        
        if not config:
            return motor_positions
            
        joint_names = list(degrees.keys())
        
        for i, motor in enumerate(config.motors):
            if i < len(joint_names):
                joint_name = joint_names[i]
                angle = degrees[joint_name]
                
                # Convert degrees to motor position (0-4095 range typically)
                # This is a simplified conversion - real implementation would use
                # proper kinematics and motor specifications
                normalized_angle = (angle + 180) / 360  # Normalize to 0-1
                motor_position = int(
                    motor.min_position + 
                    normalized_angle * (motor.max_position - motor.min_position)
                )
                
                # Apply direction and offset
                motor_position = motor.position_offset + (motor_position * motor.direction)
                
                # Clamp to limits
                motor_position = max(motor.min_position, 
                                   min(motor.max_position, motor_position))
                
                motor_positions[motor.motor_id] = motor_position
                
        return motor_positions
        
    def _validate_positions(self, positions: Dict[int, int], arm: str) -> bool:
        """Validate motor positions are within safe limits."""
        config = self.leader_config if arm == "leader" else self.follower_config
        
        if not config:
            return False
            
        for motor in config.motors:
            if motor.motor_id in positions:
                position = positions[motor.motor_id]
                if position < motor.min_position or position > motor.max_position:
                    logger.warning(
                        f"Motor {motor.motor_id} position {position} outside limits "
                        f"[{motor.min_position}, {motor.max_position}]"
                    )
                    return False
                    
        return True
        
    def _wait_for_movement_completion(self, arm: str, timeout: float = 5.0) -> bool:
        """Wait for movement to complete."""
        start_time = time.time()
        controller = self.leader_controller if arm == "leader" else self.follower_controller
        config = self.leader_config if arm == "leader" else self.follower_config
        
        if not controller or not config:
            return False
            
        while time.time() - start_time < timeout:
            all_stopped = True
            
            for motor in config.motors:
                status = controller.get_motor_status(motor.motor_id)
                if status and status.is_moving:
                    all_stopped = False
                    break
                    
            if all_stopped:
                return True
                
            time.sleep(0.1)
            
        logger.warning(f"Movement timeout for {arm}")
        return False
        
    def _get_all_motor_status(self) -> Dict[str, List[MotorStatus]]:
        """Get status for all motors."""
        status = {"leader": [], "follower": []}
        
        # Get leader status
        for motor in self.leader_config.motors:
            motor_status = self.leader_controller.get_motor_status(motor.motor_id)
            if motor_status:
                status["leader"].append(motor_status)
                
        # Get follower status if available
        if self.follower_controller and self.follower_config:
            for motor in self.follower_config.motors:
                motor_status = self.follower_controller.get_motor_status(motor.motor_id)
                if motor_status:
                    status["follower"].append(motor_status)
                    
        return status
        
    def _start_monitoring(self) -> None:
        """Start monitoring thread."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
    def _stop_monitoring(self) -> None:
        """Stop monitoring thread."""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
            
    def _monitoring_loop(self) -> None:
        """Continuous monitoring of robot status."""
        while self.is_monitoring:
            try:
                if self.state == RobotState.CONNECTED:
                    # Check motor statuses
                    motor_statuses = self._get_all_motor_status()
                    
                    # Check for errors
                    for arm_statuses in motor_statuses.values():
                        for motor_status in arm_statuses:
                            if motor_status.has_error:
                                logger.warning(
                                    f"Motor {motor_status.id} error: {motor_status.error_code}"
                                )
                                
                            # Check temperature
                            if motor_status.temperature > 70:  # 70°C warning threshold
                                logger.warning(
                                    f"Motor {motor_status.id} high temperature: {motor_status.temperature}°C"
                                )
                                
                    # Create status update
                    if self.status_callback:
                        robot_status = RobotStatus(
                            state=self.state,
                            timestamp=time.time(),
                            motors=[status for statuses in motor_statuses.values() for status in statuses]
                        )
                        self.status_callback(robot_status)
                        
                time.sleep(0.1)  # 10Hz monitoring rate
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(1.0)


def create_default_so101_config(port: str, name: str = "SO-101") -> RobotConfig:
    """Create default configuration for SO-101 robot arm."""
    motors = [
        MotorConfig(motor_id=1, model=MotorModel.XM430_W350, min_position=0, max_position=4095),
        MotorConfig(motor_id=2, model=MotorModel.XM430_W350, min_position=0, max_position=4095),
        MotorConfig(motor_id=3, model=MotorModel.XM430_W350, min_position=0, max_position=4095),
        MotorConfig(motor_id=4, model=MotorModel.XM430_W350, min_position=0, max_position=4095),
        MotorConfig(motor_id=5, model=MotorModel.XM430_W350, min_position=0, max_position=4095),
        MotorConfig(motor_id=6, model=MotorModel.XM430_W350, min_position=0, max_position=4095),  # Gripper
    ]
    
    return RobotConfig(
        name=name,
        port=port,
        motors=motors,
        workspace_limits={
            "x": [-300, 300],
            "y": [-300, 300],
            "z": [0, 400]
        }
    )


async def main():
    """Example usage of the RobotController."""
    
    def status_callback(status: RobotStatus):
        if status.motors:
            temps = [m.temperature for m in status.motors]
            avg_temp = sum(temps) / len(temps)
            console.print(f"[dim]Robot status: {status.state.value}, avg temp: {avg_temp:.1f}°C")
    
    # Create configurations
    leader_config = create_default_so101_config("/dev/tty.usbserial-FT1234", "Leader")
    follower_config = create_default_so101_config("/dev/tty.usbserial-FT5678", "Follower")
    
    # Create controller
    robot = RobotController(leader_config, follower_config)
    robot.set_status_callback(status_callback)
    
    # Connect
    if robot.connect("/dev/tty.usbserial-FT1234", "/dev/tty.usbserial-FT5678"):
        console.print("[green]✓ Robot connected")
        
        # Move to home position
        robot.home_position("leader")
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Move to ready position
        robot.ready_position("leader")
        
        # Wait a bit more
        await asyncio.sleep(2)
        
    else:
        console.print("[red]✗ Failed to connect to robot")
        
    # Cleanup
    robot.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())