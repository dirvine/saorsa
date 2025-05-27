#!/usr/bin/env python3
"""
Visual Feedback and Overlay System for Saorse Robot System

This module provides real-time visual feedback including object detection overlays,
robot status visualization, and augmented reality features for the robot workspace.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import cv2
from rich.console import Console
from PIL import Image, ImageDraw, ImageFont

# Local imports
from object_detector import Detection, ObjectDetector
from robot_controller_m3 import RobotController
from utils.safety_monitor import SafetyMonitor

console = Console()
logger = logging.getLogger(__name__)


class WorkspacePoint(NamedTuple):
    """3D point in robot workspace."""
    x: float
    y: float
    z: float
    confidence: float = 1.0


@dataclass
class OverlayConfig:
    """Visual overlay configuration."""
    show_detections: bool = True
    show_robot_status: bool = True
    show_workspace_bounds: bool = True
    show_safety_zones: bool = True
    show_trajectory: bool = False
    detection_box_thickness: int = 2
    text_scale: float = 0.6
    text_thickness: int = 1
    overlay_alpha: float = 0.7
    fps_display: bool = True
    coordinate_display: bool = True
    enable_3d_projection: bool = True


@dataclass
class VisualizationState:
    """Current visualization state."""
    current_detections: List[Detection] = field(default_factory=list)
    robot_position: Optional[Tuple[float, float, float]] = None
    robot_status: str = "Unknown"
    safety_status: str = "OK"
    workspace_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
    trajectory_points: List[WorkspacePoint] = field(default_factory=list)
    fps: float = 0.0
    processing_time: float = 0.0


class VisualFeedback:
    """
    Visual feedback system for real-time robot operation monitoring.
    
    Provides augmented reality overlays showing:
    - Object detection results
    - Robot arm position and status
    - Workspace boundaries and safety zones
    - Real-time performance metrics
    """
    
    def __init__(self, config: Optional[OverlayConfig] = None):
        self.config = config or OverlayConfig()
        
        # State management
        self.state = VisualizationState()
        self.last_update_time = time.time()
        
        # Display settings
        self.colors = self._initialize_colors()
        self.fonts = self._initialize_fonts()
        
        # Camera calibration (will be loaded from config)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.workspace_to_camera_transform = None
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_history = 60
        
    def _initialize_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Initialize color scheme for overlays."""
        
        return {
            # Detection colors
            'detection_box': (0, 255, 0),
            'detection_text': (255, 255, 255),
            'detection_bg': (0, 128, 0),
            
            # Robot colors
            'robot_position': (255, 0, 0),
            'robot_trajectory': (255, 165, 0),
            'robot_target': (255, 255, 0),
            
            # Workspace colors
            'workspace_bounds': (0, 0, 255),
            'safety_zone_ok': (0, 255, 0),
            'safety_zone_warning': (255, 255, 0),
            'safety_zone_danger': (255, 0, 0),
            
            # UI colors
            'text_primary': (255, 255, 255),
            'text_secondary': (200, 200, 200),
            'background': (0, 0, 0),
            'overlay_bg': (0, 0, 0),
            
            # Status colors
            'status_ok': (0, 255, 0),
            'status_warning': (255, 255, 0),
            'status_error': (255, 0, 0),
        }
        
    def _initialize_fonts(self) -> Dict[str, Any]:
        """Initialize fonts for text rendering."""
        
        # OpenCV uses Hershey fonts
        return {
            'default': cv2.FONT_HERSHEY_SIMPLEX,
            'small': cv2.FONT_HERSHEY_SIMPLEX,
            'large': cv2.FONT_HERSHEY_DUPLEX,
            'mono': cv2.FONT_HERSHEY_COMPLEX_SMALL,
        }
        
    def update_detections(self, detections: List[Detection]):
        """Update current object detections."""
        self.state.current_detections = detections
        self.last_update_time = time.time()
        
    def update_robot_state(self, position: Optional[Tuple[float, float, float]], 
                          status: str):
        """Update robot position and status."""
        self.state.robot_position = position
        self.state.robot_status = status
        
    def update_safety_state(self, safety_status: str, 
                           workspace_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None):
        """Update safety monitoring state."""
        self.state.safety_status = safety_status
        if workspace_bounds:
            self.state.workspace_bounds = workspace_bounds
            
    def add_trajectory_point(self, point: WorkspacePoint):
        """Add point to robot trajectory visualization."""
        self.state.trajectory_points.append(point)
        
        # Limit trajectory history
        max_trajectory_points = 100
        if len(self.state.trajectory_points) > max_trajectory_points:
            self.state.trajectory_points = self.state.trajectory_points[-max_trajectory_points:]
            
    def clear_trajectory(self):
        """Clear trajectory visualization."""
        self.state.trajectory_points = []
        
    def render_frame(self, frame: np.ndarray) -> np.ndarray:
        """Render all overlays on the input frame."""
        
        start_time = time.time()
        
        # Create overlay frame
        overlay_frame = frame.copy()
        
        # Draw detections
        if self.config.show_detections and self.state.current_detections:
            overlay_frame = self._draw_detections(overlay_frame)
            
        # Draw robot status
        if self.config.show_robot_status:
            overlay_frame = self._draw_robot_status(overlay_frame)
            
        # Draw workspace bounds
        if self.config.show_workspace_bounds and self.state.workspace_bounds:
            overlay_frame = self._draw_workspace_bounds(overlay_frame)
            
        # Draw safety zones
        if self.config.show_safety_zones:
            overlay_frame = self._draw_safety_zones(overlay_frame)
            
        # Draw trajectory
        if self.config.show_trajectory and self.state.trajectory_points:
            overlay_frame = self._draw_trajectory(overlay_frame)
            
        # Draw performance info
        if self.config.fps_display:
            overlay_frame = self._draw_performance_info(overlay_frame)
            
        # Draw coordinate system
        if self.config.coordinate_display:
            overlay_frame = self._draw_coordinate_system(overlay_frame)
            
        # Update performance metrics
        processing_time = time.time() - start_time
        self.state.processing_time = processing_time
        self._update_fps()
        
        return overlay_frame
        
    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw object detection overlays."""
        
        for detection in self.state.current_detections:
            x, y, w, h = detection.bbox
            
            # Determine color based on object class
            color = self._get_detection_color(detection.class_name)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.config.detection_box_thickness)
            
            # Draw center point
            center_x, center_y = detection.center
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
            
            # Draw confidence bar
            bar_width = min(w, 100)
            bar_height = 6
            bar_x = x
            bar_y = y - 15
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         self.colors['overlay_bg'], -1)
            
            # Confidence bar
            conf_width = int(bar_width * detection.confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), 
                         color, -1)
            
            # Draw label with confidence
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            if self.config.enable_3d_projection:
                # Add 3D position if available
                world_pos = self._pixel_to_world(detection.center)
                if world_pos:
                    label += f" ({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f})"
                    
            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.fonts['default'], self.config.text_scale, self.config.text_thickness
            )
            
            # Text background
            text_x = x
            text_y = y - 25
            cv2.rectangle(frame, (text_x, text_y - text_height - baseline), 
                         (text_x + text_width, text_y + baseline), 
                         self.colors['detection_bg'], -1)
            
            # Text
            cv2.putText(frame, label, (text_x, text_y), 
                       self.fonts['default'], self.config.text_scale, 
                       self.colors['detection_text'], self.config.text_thickness)
                       
        return frame
        
    def _draw_robot_status(self, frame: np.ndarray) -> np.ndarray:
        """Draw robot status information."""
        
        height, width = frame.shape[:2]
        
        # Status panel background
        panel_height = 120
        panel_width = 300
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     self.colors['overlay_bg'], -1)
        frame = cv2.addWeighted(frame, 1 - self.config.overlay_alpha, 
                               overlay, self.config.overlay_alpha, 0)
        
        # Status text
        status_lines = [
            f"Robot Status: {self.state.robot_status}",
            f"Safety: {self.state.safety_status}",
        ]
        
        if self.state.robot_position:
            x, y, z = self.state.robot_position
            status_lines.append(f"Position: ({x:.1f}, {y:.1f}, {z:.1f})")
        else:
            status_lines.append("Position: Unknown")
            
        # Draw status lines
        line_height = 25
        for i, line in enumerate(status_lines):
            text_y = panel_y + 20 + i * line_height
            
            # Determine color based on status
            if "Error" in line or "DANGER" in line:
                text_color = self.colors['status_error']
            elif "Warning" in line or "WARNING" in line:
                text_color = self.colors['status_warning']
            else:
                text_color = self.colors['status_ok']
                
            cv2.putText(frame, line, (panel_x + 10, text_y), 
                       self.fonts['default'], self.config.text_scale, 
                       text_color, self.config.text_thickness)
                       
        return frame
        
    def _draw_workspace_bounds(self, frame: np.ndarray) -> np.ndarray:
        """Draw robot workspace boundaries."""
        
        if not self.state.workspace_bounds:
            return frame
            
        # Project workspace bounds to image coordinates
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.state.workspace_bounds
        
        # Define workspace corners in 3D
        corners_3d = [
            (x_min, y_min, z_min), (x_max, y_min, z_min),
            (x_max, y_max, z_min), (x_min, y_max, z_min),
            (x_min, y_min, z_max), (x_max, y_min, z_max),
            (x_max, y_max, z_max), (x_min, y_max, z_max),
        ]
        
        # Project to 2D if camera calibration is available
        if self.camera_matrix is not None:
            corners_2d = []
            for corner in corners_3d:
                pixel_pos = self._world_to_pixel(corner)
                if pixel_pos:
                    corners_2d.append(pixel_pos)
                    
            if len(corners_2d) == 8:
                # Draw workspace wireframe
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                    (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
                ]
                
                for start_idx, end_idx in edges:
                    start_point = tuple(map(int, corners_2d[start_idx]))
                    end_point = tuple(map(int, corners_2d[end_idx]))
                    cv2.line(frame, start_point, end_point, 
                            self.colors['workspace_bounds'], 2)
        else:
            # Simple 2D projection (assuming top-down view)
            height, width = frame.shape[:2]
            
            # Map workspace to image coordinates (simplified)
            margin = 50
            img_x_min = margin
            img_x_max = width - margin
            img_y_min = margin
            img_y_max = height - margin
            
            # Draw workspace rectangle
            cv2.rectangle(frame, (img_x_min, img_y_min), (img_x_max, img_y_max), 
                         self.colors['workspace_bounds'], 2)
            
            # Add workspace labels
            cv2.putText(frame, "Workspace", (img_x_min, img_y_min - 10), 
                       self.fonts['default'], self.config.text_scale, 
                       self.colors['workspace_bounds'], self.config.text_thickness)
                       
        return frame
        
    def _draw_safety_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw safety zone indicators."""
        
        # Simple safety zone visualization - could be enhanced with actual zone data
        height, width = frame.shape[:2]
        
        # Safety status indicator
        status_color = self.colors['status_ok']
        if self.state.safety_status == "WARNING":
            status_color = self.colors['status_warning']
        elif self.state.safety_status == "DANGER":
            status_color = self.colors['status_error']
            
        # Draw safety indicator
        indicator_size = 20
        indicator_x = 10
        indicator_y = height - 30
        
        cv2.circle(frame, (indicator_x + indicator_size//2, indicator_y - indicator_size//2), 
                  indicator_size//2, status_color, -1)
        
        cv2.putText(frame, f"Safety: {self.state.safety_status}", 
                   (indicator_x + indicator_size + 10, indicator_y), 
                   self.fonts['default'], self.config.text_scale, 
                   status_color, self.config.text_thickness)
                   
        return frame
        
    def _draw_trajectory(self, frame: np.ndarray) -> np.ndarray:
        """Draw robot trajectory visualization."""
        
        if len(self.state.trajectory_points) < 2:
            return frame
            
        # Convert trajectory points to pixel coordinates
        pixel_points = []
        for point in self.state.trajectory_points:
            pixel_pos = self._world_to_pixel((point.x, point.y, point.z))
            if pixel_pos:
                pixel_points.append(pixel_pos)
                
        # Draw trajectory lines
        for i in range(1, len(pixel_points)):
            start_point = tuple(map(int, pixel_points[i-1]))
            end_point = tuple(map(int, pixel_points[i]))
            
            # Fade older trajectory points
            alpha = min(1.0, i / len(pixel_points))
            color = tuple(int(c * alpha) for c in self.colors['robot_trajectory'])
            
            cv2.line(frame, start_point, end_point, color, 2)
            
        # Draw trajectory points
        for i, pixel_point in enumerate(pixel_points):
            point_pos = tuple(map(int, pixel_point))
            alpha = min(1.0, i / len(pixel_points))
            color = tuple(int(c * alpha) for c in self.colors['robot_trajectory'])
            cv2.circle(frame, point_pos, 3, color, -1)
            
        return frame
        
    def _draw_performance_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS and performance information."""
        
        height, width = frame.shape[:2]
        
        # Performance text
        perf_lines = [
            f"FPS: {self.state.fps:.1f}",
            f"Processing: {self.state.processing_time*1000:.1f}ms",
            f"Detections: {len(self.state.current_detections)}",
        ]
        
        # Draw performance info
        line_height = 20
        start_y = height - len(perf_lines) * line_height - 10
        
        for i, line in enumerate(perf_lines):
            text_y = start_y + i * line_height
            cv2.putText(frame, line, (10, text_y), 
                       self.fonts['mono'], self.config.text_scale * 0.8, 
                       self.colors['text_secondary'], self.config.text_thickness)
                       
        return frame
        
    def _draw_coordinate_system(self, frame: np.ndarray) -> np.ndarray:
        """Draw coordinate system reference."""
        
        height, width = frame.shape[:2]
        
        # Draw coordinate axes in corner
        axis_length = 50
        axis_x = width - 80
        axis_y = height - 80
        
        # X-axis (red)
        cv2.arrowedLine(frame, (axis_x, axis_y), (axis_x + axis_length, axis_y), 
                       (0, 0, 255), 2, tipLength=0.2)
        cv2.putText(frame, "X", (axis_x + axis_length + 5, axis_y + 5), 
                   self.fonts['default'], 0.5, (0, 0, 255), 1)
        
        # Y-axis (green)
        cv2.arrowedLine(frame, (axis_x, axis_y), (axis_x, axis_y - axis_length), 
                       (0, 255, 0), 2, tipLength=0.2)
        cv2.putText(frame, "Y", (axis_x - 10, axis_y - axis_length - 5), 
                   self.fonts['default'], 0.5, (0, 255, 0), 1)
        
        # Z-axis (blue) - diagonal to simulate depth
        cv2.arrowedLine(frame, (axis_x, axis_y), (axis_x - 25, axis_y - 25), 
                       (255, 0, 0), 2, tipLength=0.2)
        cv2.putText(frame, "Z", (axis_x - 35, axis_y - 30), 
                   self.fonts['default'], 0.5, (255, 0, 0), 1)
                   
        return frame
        
    def _get_detection_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for detection based on class."""
        
        # Color mapping for common objects
        color_map = {
            'person': (255, 0, 0),
            'cup': (0, 255, 255),
            'bottle': (255, 255, 0),
            'apple': (0, 255, 0),
            'orange': (0, 165, 255),
            'book': (128, 0, 128),
            'laptop': (255, 192, 203),
            'cell phone': (0, 0, 255),
        }
        
        if class_name in color_map:
            return color_map[class_name]
        else:
            # Generate consistent color based on class name hash
            hash_val = hash(class_name) % 16777215  # Max RGB value
            return (
                (hash_val >> 16) & 255,
                (hash_val >> 8) & 255,
                hash_val & 255
            )
            
    def _pixel_to_world(self, pixel_pos: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:
        """Convert pixel coordinates to world coordinates."""
        
        # This would require camera calibration data
        # For now, return None to indicate no conversion available
        return None
        
    def _world_to_pixel(self, world_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
        """Convert world coordinates to pixel coordinates."""
        
        # This would require camera calibration data
        # For now, return None to indicate no conversion available
        return None
        
    def _update_fps(self):
        """Update FPS calculation."""
        
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frame times
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times = self.frame_times[-self.max_frame_history:]
            
        # Calculate FPS
        if len(self.frame_times) > 1:
            time_span = self.frame_times[-1] - self.frame_times[0]
            if time_span > 0:
                self.state.fps = (len(self.frame_times) - 1) / time_span
                
    def save_annotated_frame(self, frame: np.ndarray, filename: str):
        """Save annotated frame to file."""
        
        try:
            annotated_frame = self.render_frame(frame)
            cv2.imwrite(filename, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            console.print(f"[green]✓ Saved annotated frame to {filename}[/green]")
            return True
        except Exception as e:
            logger.error(f"Failed to save annotated frame: {e}")
            return False
            
    def get_visualization_stats(self) -> Dict[str, Any]:
        """Get visualization performance statistics."""
        
        return {
            "fps": self.state.fps,
            "processing_time": self.state.processing_time,
            "active_detections": len(self.state.current_detections),
            "trajectory_points": len(self.state.trajectory_points),
            "robot_status": self.state.robot_status,
            "safety_status": self.state.safety_status,
            "config": {
                "show_detections": self.config.show_detections,
                "show_robot_status": self.config.show_robot_status,
                "show_workspace_bounds": self.config.show_workspace_bounds,
                "enable_3d_projection": self.config.enable_3d_projection,
            }
        }


async def main():
    """Example usage of VisualFeedback."""
    
    console.print("[blue]Saorse Visual Feedback Demo[/blue]\n")
    
    # Create visual feedback system
    config = OverlayConfig(
        show_detections=True,
        show_robot_status=True,
        show_workspace_bounds=True,
        fps_display=True
    )
    
    visual_feedback = VisualFeedback(config)
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add some synthetic objects for testing
    cv2.rectangle(test_frame, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(test_frame, (400, 300), 50, (0, 255, 0), -1)
    
    # Mock detection data
    from object_detector import Detection
    test_detections = [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=(100, 100, 100, 100),
            center=(150, 150),
            area=10000
        ),
        Detection(
            class_id=47,
            class_name="cup",
            confidence=0.72,
            bbox=(350, 250, 100, 100),
            center=(400, 300),
            area=10000
        )
    ]
    
    # Update visual feedback state
    visual_feedback.update_detections(test_detections)
    visual_feedback.update_robot_state((250.5, 180.2, 120.0), "Operating")
    visual_feedback.update_safety_state("OK", ((-300, 300), (-300, 300), (0, 400)))
    
    # Add some trajectory points
    trajectory_points = [
        WorkspacePoint(200, 150, 100),
        WorkspacePoint(220, 160, 105),
        WorkspacePoint(240, 170, 110),
        WorkspacePoint(250, 180, 120),
    ]
    
    for point in trajectory_points:
        visual_feedback.add_trajectory_point(point)
        
    # Render annotated frame
    console.print("[blue]Rendering annotated frame...[/blue]")
    
    annotated_frame = visual_feedback.render_frame(test_frame)
    
    # Save result
    output_path = "/tmp/saorse_visual_feedback_demo.jpg"
    if visual_feedback.save_annotated_frame(test_frame, output_path):
        console.print(f"[green]✓ Demo frame saved to {output_path}[/green]")
        
    # Display stats
    stats = visual_feedback.get_visualization_stats()
    console.print(f"[blue]Visualization stats:[/blue]")
    for key, value in stats.items():
        if isinstance(value, dict):
            console.print(f"  {key}:")
            for sub_key, sub_value in value.items():
                console.print(f"    {sub_key}: {sub_value}")
        else:
            console.print(f"  {key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())