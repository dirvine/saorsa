#!/usr/bin/env python3
"""
Safety Monitor for Saorse Robot System

This module provides comprehensive safety monitoring including workspace limits,
velocity constraints, temperature monitoring, and emergency stop functionality.
"""

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety alert levels."""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafetyEvent(Enum):
    """Types of safety events."""
    WORKSPACE_VIOLATION = "workspace_violation"
    VELOCITY_EXCEEDED = "velocity_exceeded"
    TEMPERATURE_HIGH = "temperature_high"
    MOTOR_ERROR = "motor_error"
    COMMUNICATION_LOST = "communication_lost"
    COLLISION_DETECTED = "collision_detected"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyAlert:
    """Safety alert information."""
    event: SafetyEvent
    level: SafetyLevel
    message: str
    timestamp: float
    motor_id: Optional[int] = None
    position: Optional[Tuple[float, float, float]] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class WorkspaceLimits:
    """3D workspace boundary limits."""
    x_min: float = -300.0
    x_max: float = 300.0
    y_min: float = -300.0
    y_max: float = 300.0
    z_min: float = 0.0
    z_max: float = 400.0
    
    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if point is within workspace."""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)
                
    def distance_to_boundary(self, x: float, y: float, z: float) -> float:
        """Calculate minimum distance to workspace boundary."""
        distances = [
            abs(x - self.x_min), abs(x - self.x_max),
            abs(y - self.y_min), abs(y - self.y_max),
            abs(z - self.z_min), abs(z - self.z_max)
        ]
        return min(distances)


@dataclass
class SafetyLimits:
    """Safety constraints and thresholds."""
    max_velocity: float = 30.0  # degrees/second
    max_acceleration: float = 100.0  # degrees/second²
    max_temperature: float = 70.0  # Celsius
    warning_temperature: float = 60.0  # Celsius
    max_current: float = 500.0  # mA
    warning_current: float = 400.0  # mA
    min_voltage: float = 11.0  # Volts
    workspace_buffer: float = 10.0  # mm buffer from workspace limits
    collision_force_threshold: float = 50.0  # N (future feature)
    emergency_stop_words: List[str] = field(default_factory=lambda: [
        "stop", "halt", "emergency", "cease", "freeze"
    ])


class SafetyMonitor:
    """
    Comprehensive safety monitoring system.
    
    Monitors robot state for safety violations and provides
    alerts, warnings, and emergency stop functionality.
    """
    
    def __init__(self, workspace_limits: WorkspaceLimits, safety_limits: SafetyLimits):
        self.workspace_limits = workspace_limits
        self.safety_limits = safety_limits
        
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Safety state
        self.current_safety_level = SafetyLevel.SAFE
        self.active_alerts: List[SafetyAlert] = []
        self.alert_history: List[SafetyAlert] = []
        
        # Callbacks
        self.alert_callbacks: List[Callable[[SafetyAlert], None]] = []
        self.emergency_stop_callback: Optional[Callable[[], None]] = None
        
        # Monitoring data
        self.last_positions: Dict[int, Tuple[float, float, float]] = {}
        self.last_position_time: Dict[int, float] = {}
        self.velocity_history: Dict[int, List[Tuple[float, float]]] = {}  # (velocity, timestamp)
        
        # Voice command monitoring
        self.last_voice_command_time = 0
        self.voice_timeout = 10.0  # seconds
        
    def start_monitoring(self) -> None:
        """Start safety monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        console.print("[green]🛡️  Safety monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop safety monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
            
        console.print("[yellow]🛡️  Safety monitoring stopped")
        
    def add_alert_callback(self, callback: Callable[[SafetyAlert], None]) -> None:
        """Add callback for safety alerts."""
        self.alert_callbacks.append(callback)
        
    def set_emergency_stop_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for emergency stop activation."""
        self.emergency_stop_callback = callback
        
    def check_workspace_violation(self, position: Tuple[float, float, float], motor_id: int = 0) -> Optional[SafetyAlert]:
        """Check if position violates workspace limits."""
        x, y, z = position
        
        if not self.workspace_limits.contains_point(x, y, z):
            return SafetyAlert(
                event=SafetyEvent.WORKSPACE_VIOLATION,
                level=SafetyLevel.CRITICAL,
                message=f"Position ({x:.1f}, {y:.1f}, {z:.1f}) outside workspace",
                timestamp=time.time(),
                motor_id=motor_id,
                position=position
            )
            
        # Check buffer zone
        distance = self.workspace_limits.distance_to_boundary(x, y, z)
        if distance < self.safety_limits.workspace_buffer:
            return SafetyAlert(
                event=SafetyEvent.WORKSPACE_VIOLATION,
                level=SafetyLevel.WARNING,
                message=f"Position near workspace boundary (distance: {distance:.1f}mm)",
                timestamp=time.time(),
                motor_id=motor_id,
                position=position,
                value=distance,
                threshold=self.safety_limits.workspace_buffer
            )
            
        return None
        
    def check_velocity_limits(self, motor_id: int, current_position: float, timestamp: float) -> Optional[SafetyAlert]:
        """Check if motor velocity exceeds limits."""
        if motor_id not in self.last_positions or motor_id not in self.last_position_time:
            # First measurement, store and return
            self.last_positions[motor_id] = (current_position, 0, 0)
            self.last_position_time[motor_id] = timestamp
            return None
            
        last_position = self.last_positions[motor_id][0]  # Assuming 1D position for motors
        last_time = self.last_position_time[motor_id]
        
        dt = timestamp - last_time
        if dt <= 0:
            return None
            
        # Calculate velocity (degrees/second)
        velocity = abs(current_position - last_position) / dt
        
        # Store velocity history
        if motor_id not in self.velocity_history:
            self.velocity_history[motor_id] = []
        self.velocity_history[motor_id].append((velocity, timestamp))
        
        # Keep only recent history (last 5 seconds)
        cutoff_time = timestamp - 5.0
        self.velocity_history[motor_id] = [
            (v, t) for v, t in self.velocity_history[motor_id] if t > cutoff_time
        ]
        
        # Update stored values
        self.last_positions[motor_id] = (current_position, 0, 0)
        self.last_position_time[motor_id] = timestamp
        
        # Check velocity limit
        if velocity > self.safety_limits.max_velocity:
            return SafetyAlert(
                event=SafetyEvent.VELOCITY_EXCEEDED,
                level=SafetyLevel.CRITICAL,
                message=f"Motor {motor_id} velocity {velocity:.1f}°/s exceeds limit",
                timestamp=timestamp,
                motor_id=motor_id,
                value=velocity,
                threshold=self.safety_limits.max_velocity
            )
            
        return None
        
    def check_temperature_limits(self, motor_id: int, temperature: float) -> Optional[SafetyAlert]:
        """Check motor temperature limits."""
        if temperature >= self.safety_limits.max_temperature:
            return SafetyAlert(
                event=SafetyEvent.TEMPERATURE_HIGH,
                level=SafetyLevel.CRITICAL,
                message=f"Motor {motor_id} temperature {temperature}°C exceeds maximum",
                timestamp=time.time(),
                motor_id=motor_id,
                value=temperature,
                threshold=self.safety_limits.max_temperature
            )
        elif temperature >= self.safety_limits.warning_temperature:
            return SafetyAlert(
                event=SafetyEvent.TEMPERATURE_HIGH,
                level=SafetyLevel.WARNING,
                message=f"Motor {motor_id} temperature {temperature}°C is high",
                timestamp=time.time(),
                motor_id=motor_id,
                value=temperature,
                threshold=self.safety_limits.warning_temperature
            )
            
        return None
        
    def check_current_limits(self, motor_id: int, current: float) -> Optional[SafetyAlert]:
        """Check motor current limits."""
        if current >= self.safety_limits.max_current:
            return SafetyAlert(
                event=SafetyEvent.MOTOR_ERROR,
                level=SafetyLevel.CRITICAL,
                message=f"Motor {motor_id} current {current}mA exceeds maximum",
                timestamp=time.time(),
                motor_id=motor_id,
                value=current,
                threshold=self.safety_limits.max_current
            )
        elif current >= self.safety_limits.warning_current:
            return SafetyAlert(
                event=SafetyEvent.MOTOR_ERROR,
                level=SafetyLevel.WARNING,
                message=f"Motor {motor_id} current {current}mA is high",
                timestamp=time.time(),
                motor_id=motor_id,
                value=current,
                threshold=self.safety_limits.warning_current
            )
            
        return None
        
    def check_voltage_limits(self, voltage: float) -> Optional[SafetyAlert]:
        """Check system voltage limits."""
        if voltage < self.safety_limits.min_voltage:
            return SafetyAlert(
                event=SafetyEvent.MOTOR_ERROR,
                level=SafetyLevel.WARNING,
                message=f"System voltage {voltage:.1f}V is low",
                timestamp=time.time(),
                value=voltage,
                threshold=self.safety_limits.min_voltage
            )
            
        return None
        
    def check_voice_command_safety(self, command_text: str) -> Optional[SafetyAlert]:
        """Check if voice command contains emergency stop words."""
        command_lower = command_text.lower()
        
        for stop_word in self.safety_limits.emergency_stop_words:
            if stop_word in command_lower:
                return SafetyAlert(
                    event=SafetyEvent.EMERGENCY_STOP,
                    level=SafetyLevel.EMERGENCY,
                    message=f"Emergency stop triggered by voice command: '{command_text}'",
                    timestamp=time.time()
                )
                
        return None
        
    def update_motor_status(self, motor_id: int, position: float, velocity: float, 
                          current: float, temperature: float, voltage: float, 
                          has_error: bool) -> List[SafetyAlert]:
        """Update motor status and check for safety violations."""
        alerts = []
        timestamp = time.time()
        
        # Check velocity limits
        velocity_alert = self.check_velocity_limits(motor_id, position, timestamp)
        if velocity_alert:
            alerts.append(velocity_alert)
            
        # Check temperature
        temp_alert = self.check_temperature_limits(motor_id, temperature)
        if temp_alert:
            alerts.append(temp_alert)
            
        # Check current
        current_alert = self.check_current_limits(motor_id, current)
        if current_alert:
            alerts.append(current_alert)
            
        # Check voltage
        voltage_alert = self.check_voltage_limits(voltage)
        if voltage_alert:
            alerts.append(voltage_alert)
            
        # Check for motor errors
        if has_error:
            error_alert = SafetyAlert(
                event=SafetyEvent.MOTOR_ERROR,
                level=SafetyLevel.CRITICAL,
                message=f"Motor {motor_id} reported hardware error",
                timestamp=timestamp,
                motor_id=motor_id
            )
            alerts.append(error_alert)
            
        # Process alerts
        for alert in alerts:
            self._handle_alert(alert)
            
        return alerts
        
    def update_end_effector_position(self, position: Tuple[float, float, float]) -> Optional[SafetyAlert]:
        """Update end effector position and check workspace limits."""
        alert = self.check_workspace_violation(position)
        if alert:
            self._handle_alert(alert)
        return alert
        
    def process_voice_command(self, command_text: str) -> Optional[SafetyAlert]:
        """Process voice command for safety checks."""
        self.last_voice_command_time = time.time()
        
        alert = self.check_voice_command_safety(command_text)
        if alert:
            self._handle_alert(alert)
            
        return alert
        
    def get_current_safety_level(self) -> SafetyLevel:
        """Get current overall safety level."""
        return self.current_safety_level
        
    def get_active_alerts(self) -> List[SafetyAlert]:
        """Get list of currently active alerts."""
        return self.active_alerts.copy()
        
    def clear_alert(self, alert: SafetyAlert) -> None:
        """Clear an active alert."""
        if alert in self.active_alerts:
            self.active_alerts.remove(alert)
            self._update_safety_level()
            
    def clear_all_alerts(self) -> None:
        """Clear all active alerts."""
        self.active_alerts.clear()
        self.current_safety_level = SafetyLevel.SAFE
        
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get safety monitoring statistics."""
        stats = {
            "current_level": self.current_safety_level.value,
            "active_alerts_count": len(self.active_alerts),
            "total_alerts_count": len(self.alert_history),
            "monitoring_active": self.is_monitoring,
            "last_voice_command_age": time.time() - self.last_voice_command_time,
        }
        
        # Alert type distribution
        alert_types = {}
        for alert in self.alert_history:
            event_type = alert.event.value
            alert_types[event_type] = alert_types.get(event_type, 0) + 1
        stats["alert_distribution"] = alert_types
        
        # Average motor velocities
        avg_velocities = {}
        for motor_id, history in self.velocity_history.items():
            if history:
                velocities = [v for v, _ in history]
                avg_velocities[motor_id] = sum(velocities) / len(velocities)
        stats["average_velocities"] = avg_velocities
        
        return stats
        
    def _handle_alert(self, alert: SafetyAlert) -> None:
        """Handle a new safety alert."""
        # Add to active alerts if not already present
        if alert not in self.active_alerts:
            self.active_alerts.append(alert)
            
        # Add to history
        self.alert_history.append(alert)
        
        # Update safety level
        self._update_safety_level()
        
        # Trigger emergency stop for emergency level alerts
        if alert.level == SafetyLevel.EMERGENCY and self.emergency_stop_callback:
            self.emergency_stop_callback()
            
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
                
        # Log alert
        level_colors = {
            SafetyLevel.SAFE: "green",
            SafetyLevel.WARNING: "yellow",
            SafetyLevel.CRITICAL: "orange",
            SafetyLevel.EMERGENCY: "red bold"
        }
        
        color = level_colors.get(alert.level, "white")
        console.print(f"[{color}]🚨 {alert.level.value.upper()}: {alert.message}")
        
    def _update_safety_level(self) -> None:
        """Update current safety level based on active alerts."""
        if not self.active_alerts:
            self.current_safety_level = SafetyLevel.SAFE
            return
            
        # Find highest level among active alerts
        max_level = SafetyLevel.SAFE
        for alert in self.active_alerts:
            if alert.level.value == "emergency":
                max_level = SafetyLevel.EMERGENCY
                break
            elif alert.level.value == "critical" and max_level.value != "emergency":
                max_level = SafetyLevel.CRITICAL
            elif alert.level.value == "warning" and max_level.value not in ["emergency", "critical"]:
                max_level = SafetyLevel.WARNING
                
        self.current_safety_level = max_level
        
    def _monitoring_loop(self) -> None:
        """Main safety monitoring loop."""
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Check for communication timeout
                if (current_time - self.last_voice_command_time > self.voice_timeout and 
                    self.last_voice_command_time > 0):
                    
                    # This could be extended to check for robot communication timeout
                    pass
                    
                # Clean up old alerts (older than 1 hour)
                cutoff_time = current_time - 3600
                self.alert_history = [
                    alert for alert in self.alert_history 
                    if alert.timestamp > cutoff_time
                ]
                
                # Auto-clear warning level alerts after some time
                alerts_to_clear = []
                for alert in self.active_alerts:
                    if (alert.level == SafetyLevel.WARNING and 
                        current_time - alert.timestamp > 30):  # 30 seconds
                        alerts_to_clear.append(alert)
                        
                for alert in alerts_to_clear:
                    self.clear_alert(alert)
                    
                time.sleep(1.0)  # 1Hz monitoring rate
                
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                time.sleep(1.0)


def create_default_safety_config() -> Tuple[WorkspaceLimits, SafetyLimits]:
    """Create default safety configuration for SO-101."""
    workspace = WorkspaceLimits(
        x_min=-300.0, x_max=300.0,
        y_min=-300.0, y_max=300.0,
        z_min=0.0, z_max=400.0
    )
    
    safety = SafetyLimits(
        max_velocity=30.0,
        max_acceleration=100.0,
        max_temperature=70.0,
        warning_temperature=60.0,
        max_current=500.0,
        warning_current=400.0,
        min_voltage=11.0,
        workspace_buffer=10.0,
        emergency_stop_words=["stop", "halt", "emergency", "cease", "freeze"]
    )
    
    return workspace, safety


async def main():
    """Example usage of SafetyMonitor."""
    
    def alert_callback(alert: SafetyAlert):
        console.print(f"[red]SAFETY ALERT: {alert.message}")
        
    def emergency_stop():
        console.print("[red bold]🚨 EMERGENCY STOP ACTIVATED 🚨")
        
    # Create safety monitor
    workspace, safety_limits = create_default_safety_config()
    monitor = SafetyMonitor(workspace, safety_limits)
    
    # Set callbacks
    monitor.add_alert_callback(alert_callback)
    monitor.set_emergency_stop_callback(emergency_stop)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some safety events
    await asyncio.sleep(1)
    
    # Test workspace violation
    monitor.update_end_effector_position((350, 0, 200))  # Outside workspace
    
    await asyncio.sleep(1)
    
    # Test emergency voice command
    monitor.process_voice_command("emergency stop now")
    
    await asyncio.sleep(1)
    
    # Test motor status
    monitor.update_motor_status(
        motor_id=1, position=100, velocity=5, current=300, 
        temperature=75, voltage=12.0, has_error=False  # High temperature
    )
    
    await asyncio.sleep(5)
    
    # Show statistics
    stats = monitor.get_safety_statistics()
    console.print(f"[blue]Safety Stats: {stats}")
    
    # Stop monitoring
    monitor.stop_monitoring()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())