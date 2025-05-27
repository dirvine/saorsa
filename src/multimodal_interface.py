#!/usr/bin/env python3
"""
Multimodal Interface for Saorse Robot System

This module integrates voice commands with visual understanding to enable
sophisticated multimodal robot control combining speech and vision.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import cv2
from rich.console import Console

# Local imports
from mac_audio_handler import MacAudioHandler
from mac_camera_handler import MacCameraHandler, CameraConfig
from object_detector import ObjectDetector, DetectionConfig, Detection
from visual_feedback import VisualFeedback, OverlayConfig
from context_manager import ContextManager
from ai_command_processor import AICommandProcessor

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class MultimodalConfig:
    """Configuration for multimodal interface."""
    # Audio settings
    enable_voice: bool = True
    voice_activation_threshold: float = 0.7
    continuous_listening: bool = True
    
    # Vision settings
    enable_vision: bool = True
    camera_resolution: Tuple[int, int] = (1280, 720)
    detection_fps: int = 10
    detection_confidence: float = 0.6
    
    # Visual feedback settings
    enable_visual_feedback: bool = True
    show_detection_overlays: bool = True
    show_spatial_grid: bool = True
    
    # Integration settings
    multimodal_fusion_enabled: bool = True
    spatial_reference_resolution: bool = True
    visual_confirmation_requests: bool = True
    
    # Performance settings
    max_processing_fps: int = 30
    frame_skip_ratio: int = 2  # Process every Nth frame for detection


@dataclass
class MultimodalCommand:
    """A command combining voice and visual information."""
    timestamp: float
    voice_text: str
    voice_confidence: float
    visual_detections: List[Detection]
    resolved_command: str
    spatial_references: Dict[str, Any]
    confidence_score: float


class SpatialReferenceResolver:
    """Resolves spatial references using vision and context."""
    
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        
    def resolve_spatial_references(self, command: str, detections: List[Detection]) -> Dict[str, Any]:
        """Resolve spatial references in a voice command using visual context."""
        
        references = {}
        command_lower = command.lower()
        
        # Handle demonstrative pronouns with spatial context
        if "that" in command_lower and detections:
            # "that object" - find object based on recent focus or largest object
            target_object = self._find_target_object(detections, preference="recent")
            if target_object:
                references["that"] = target_object.class_name
                references["that_position"] = target_object.center
                
        if "this" in command_lower and detections:
            # "this object" - find closest/most prominent object
            target_object = self._find_target_object(detections, preference="prominent")
            if target_object:
                references["this"] = target_object.class_name
                references["this_position"] = target_object.center
                
        # Handle spatial directional references
        spatial_terms = {
            "left": self._find_leftmost_object,
            "right": self._find_rightmost_object,
            "center": self._find_center_object,
            "front": self._find_front_object,
            "back": self._find_back_object,
        }
        
        for term, finder_func in spatial_terms.items():
            if term in command_lower:
                obj = finder_func(detections)
                if obj:
                    references[f"{term}_object"] = obj.class_name
                    references[f"{term}_position"] = obj.center
                    
        # Handle relational references
        if "next to" in command_lower or "beside" in command_lower:
            # Find objects that are spatially close
            reference_pairs = self._find_adjacent_objects(detections)
            references["adjacent_objects"] = reference_pairs
            
        if "between" in command_lower:
            # Find object between other objects
            middle_objects = self._find_middle_objects(detections)
            references["middle_objects"] = middle_objects
            
        # Handle size-based references
        if "big" in command_lower or "large" in command_lower:
            largest_obj = self._find_largest_object(detections)
            if largest_obj:
                references["large_object"] = largest_obj.class_name
                references["large_object_position"] = largest_obj.center
                
        if "small" in command_lower or "little" in command_lower:
            smallest_obj = self._find_smallest_object(detections)
            if smallest_obj:
                references["small_object"] = smallest_obj.class_name
                references["small_object_position"] = smallest_obj.center
                
        return references
        
    def _find_target_object(self, detections: List[Detection], preference: str = "recent") -> Optional[Detection]:
        """Find target object based on preference strategy."""
        
        if not detections:
            return None
            
        if preference == "recent":
            # Use context to find recently mentioned object
            recent_objects = self.context_manager.reference_resolver._get_recent_objects()
            for obj_name in recent_objects:
                for detection in detections:
                    if detection.class_name == obj_name:
                        return detection
                        
        elif preference == "prominent":
            # Find most prominent (largest or highest confidence) object
            return max(detections, key=lambda d: d.confidence * d.area)
            
        # Fallback to highest confidence detection
        return max(detections, key=lambda d: d.confidence)
        
    def _find_leftmost_object(self, detections: List[Detection]) -> Optional[Detection]:
        """Find leftmost object in the scene."""
        if not detections:
            return None
        return min(detections, key=lambda d: d.center[0])
        
    def _find_rightmost_object(self, detections: List[Detection]) -> Optional[Detection]:
        """Find rightmost object in the scene."""
        if not detections:
            return None
        return max(detections, key=lambda d: d.center[0])
        
    def _find_center_object(self, detections: List[Detection]) -> Optional[Detection]:
        """Find object closest to center of image."""
        if not detections:
            return None
            
        center_x, center_y = 640, 360  # Assuming 1280x720 resolution
        return min(detections, key=lambda d: 
                  np.sqrt((d.center[0] - center_x)**2 + (d.center[1] - center_y)**2))
        
    def _find_front_object(self, detections: List[Detection]) -> Optional[Detection]:
        """Find object closest to front (bottom of image)."""
        if not detections:
            return None
        return max(detections, key=lambda d: d.center[1])
        
    def _find_back_object(self, detections: List[Detection]) -> Optional[Detection]:
        """Find object closest to back (top of image)."""
        if not detections:
            return None
        return min(detections, key=lambda d: d.center[1])
        
    def _find_largest_object(self, detections: List[Detection]) -> Optional[Detection]:
        """Find largest object by area."""
        if not detections:
            return None
        return max(detections, key=lambda d: d.area)
        
    def _find_smallest_object(self, detections: List[Detection]) -> Optional[Detection]:
        """Find smallest object by area."""
        if not detections:
            return None
        return min(detections, key=lambda d: d.area)
        
    def _find_adjacent_objects(self, detections: List[Detection]) -> List[Tuple[str, str]]:
        """Find pairs of objects that are close to each other."""
        adjacent_pairs = []
        
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i+1:], i+1):
                # Calculate distance between centers
                distance = np.sqrt((det1.center[0] - det2.center[0])**2 + 
                                 (det1.center[1] - det2.center[1])**2)
                
                # If objects are close (within 200 pixels), consider them adjacent
                if distance < 200:
                    adjacent_pairs.append((det1.class_name, det2.class_name))
                    
        return adjacent_pairs
        
    def _find_middle_objects(self, detections: List[Detection]) -> List[str]:
        """Find objects that are between other objects."""
        if len(detections) < 3:
            return []
            
        middle_objects = []
        
        for i, middle_det in enumerate(detections):
            # Check if this object is between any two other objects
            for j, left_det in enumerate(detections):
                if i == j:
                    continue
                for k, right_det in enumerate(detections):
                    if i == k or j == k:
                        continue
                        
                    # Check if middle object is between left and right objects
                    if self._is_between(middle_det, left_det, right_det):
                        middle_objects.append(middle_det.class_name)
                        break
                        
        return list(set(middle_objects))  # Remove duplicates
        
    def _is_between(self, middle: Detection, left: Detection, right: Detection) -> bool:
        """Check if middle object is between left and right objects."""
        # Simple check - middle object's x-coordinate is between left and right
        left_x, middle_x, right_x = left.center[0], middle.center[0], right.center[0]
        
        # Ensure left is actually on the left
        if left_x > right_x:
            left_x, right_x = right_x, left_x
            
        return left_x < middle_x < right_x


class MultimodalInterface:
    """
    Multimodal interface combining voice commands with computer vision.
    
    Enables sophisticated robot control by understanding both spoken commands
    and visual context, allowing for natural spatial references and object manipulation.
    """
    
    def __init__(self, config: Optional[MultimodalConfig] = None):
        self.config = config or MultimodalConfig()
        
        # Component initialization
        self.audio_handler = None
        self.camera_handler = None
        self.object_detector = None
        self.visual_feedback = None
        self.context_manager = ContextManager()
        self.ai_processor = None
        self.spatial_resolver = SpatialReferenceResolver(self.context_manager)
        
        # State management
        self.is_running = False
        self.current_frame = None
        self.current_detections = []
        self.command_queue = asyncio.Queue()
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        
        # Command history
        self.command_history: List[MultimodalCommand] = []
        
    async def initialize(self) -> bool:
        """Initialize all multimodal components."""
        
        console.print("[blue]Initializing Multimodal Interface...[/blue]")
        
        try:
            # Initialize audio handler
            if self.config.enable_voice:
                self.audio_handler = MacAudioHandler()
                if not await self.audio_handler.initialize():
                    console.print("[yellow]Warning: Audio initialization failed[/yellow]")
                    self.config.enable_voice = False
                else:
                    console.print("[green]✓ Audio handler initialized[/green]")
                    
            # Initialize camera handler
            if self.config.enable_vision:
                camera_config = CameraConfig(
                    resolution=self.config.camera_resolution,
                    fps=30,
                    enable_continuity_camera=True
                )
                self.camera_handler = MacCameraHandler(camera_config)
                
                if not self.camera_handler.start_capture():
                    console.print("[yellow]Warning: Camera initialization failed[/yellow]") 
                    self.config.enable_vision = False
                else:
                    console.print("[green]✓ Camera handler initialized[/green]")
                    
            # Initialize object detector
            if self.config.enable_vision:
                detection_config = DetectionConfig(
                    model_name="facebook/detr-resnet-50",
                    confidence_threshold=self.config.detection_confidence,
                    enable_tracking=True
                )
                self.object_detector = ObjectDetector(detection_config)
                
                if not await self.object_detector.initialize():
                    console.print("[yellow]Warning: Object detector initialization failed[/yellow]")
                    self.config.enable_vision = False
                else:
                    console.print("[green]✓ Object detector initialized[/green]")
                    
            # Initialize visual feedback
            if self.config.enable_visual_feedback:
                overlay_config = OverlayConfig(
                    show_detections=self.config.show_detection_overlays,
                    show_workspace_bounds=self.config.show_spatial_grid,
                    fps_display=True
                )
                self.visual_feedback = VisualFeedback(overlay_config)
                console.print("[green]✓ Visual feedback initialized[/green]")
                
            # Initialize AI command processor
            self.ai_processor = AICommandProcessor()
            if not await self.ai_processor.initialize():
                console.print("[red]✗ AI processor initialization failed[/red]")
                return False
            else:
                console.print("[green]✓ AI processor initialized[/green]")
                
            console.print("[green]✓ Multimodal interface initialized successfully[/green]")
            return True
            
        except Exception as e:
            logger.error(f"Multimodal interface initialization failed: {e}")
            console.print(f"[red]✗ Initialization failed: {e}[/red]")
            return False
            
    async def start(self):
        """Start multimodal processing loop."""
        
        if not await self.initialize():
            return False
            
        self.is_running = True
        console.print("[blue]Starting multimodal interface...[/blue]")
        
        # Start concurrent processing tasks
        tasks = []
        
        if self.config.enable_vision:
            tasks.append(asyncio.create_task(self._vision_processing_loop()))
            
        if self.config.enable_voice:
            tasks.append(asyncio.create_task(self._voice_processing_loop()))
            
        tasks.append(asyncio.create_task(self._command_processing_loop()))
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Multimodal processing error: {e}")
        finally:
            await self.stop()
            
    async def stop(self):
        """Stop multimodal interface."""
        
        self.is_running = False
        console.print("[yellow]Stopping multimodal interface...[/yellow]")
        
        # Stop components
        if self.camera_handler:
            self.camera_handler.stop_capture()
            
        if self.object_detector:
            self.object_detector.cleanup()
            
        console.print("[yellow]Multimodal interface stopped[/yellow]")
        
    async def _vision_processing_loop(self):
        """Main vision processing loop."""
        
        frame_skip_counter = 0
        
        while self.is_running:
            try:
                # Get latest frame
                frame = self.camera_handler.get_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                    
                self.current_frame = frame
                
                # Skip frames to control processing rate
                frame_skip_counter += 1
                if frame_skip_counter % self.config.frame_skip_ratio != 0:
                    continue
                    
                # Run object detection
                detections = await self.object_detector.detect_objects_async(frame)
                self.current_detections = detections
                
                # Update context with visual information
                self.context_manager.update_visual_detections(detections, frame)
                
                # Update visual feedback
                if self.visual_feedback:
                    self.visual_feedback.update_detections(detections)
                    
                # Update performance metrics
                self._update_fps()
                
                # Control processing rate
                await asyncio.sleep(1.0 / self.config.detection_fps)
                
            except Exception as e:
                logger.error(f"Vision processing error: {e}")
                await asyncio.sleep(0.1)
                
    async def _voice_processing_loop(self):
        """Main voice processing loop."""
        
        while self.is_running:
            try:
                # Listen for voice commands
                if self.config.continuous_listening:
                    result = await self.audio_handler.listen_continuously()
                else:
                    result = await self.audio_handler.listen_once()
                    
                if result and result.get('confidence', 0) >= self.config.voice_activation_threshold:
                    voice_text = result['text']
                    voice_confidence = result['confidence']
                    
                    # Create multimodal command
                    command = MultimodalCommand(
                        timestamp=time.time(),
                        voice_text=voice_text,
                        voice_confidence=voice_confidence,
                        visual_detections=self.current_detections.copy(),
                        resolved_command="",
                        spatial_references={},
                        confidence_score=0.0
                    )
                    
                    # Queue for processing
                    await self.command_queue.put(command)
                    
            except Exception as e:
                logger.error(f"Voice processing error: {e}")
                await asyncio.sleep(0.1)
                
    async def _command_processing_loop(self):
        """Main command processing and fusion loop."""
        
        while self.is_running:
            try:
                # Get command from queue
                command = await asyncio.wait_for(self.command_queue.get(), timeout=0.1)
                
                # Process multimodal command
                await self._process_multimodal_command(command)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Command processing error: {e}")
                await asyncio.sleep(0.1)
                
    async def _process_multimodal_command(self, command: MultimodalCommand):
        """Process a multimodal command combining voice and vision."""
        
        console.print(f"[cyan]Processing: '{command.voice_text}'[/cyan]")
        
        # Resolve spatial references using vision
        if self.config.spatial_reference_resolution:
            command.spatial_references = self.spatial_resolver.resolve_spatial_references(
                command.voice_text, command.visual_detections
            )
            
        # Add command to conversation context
        self.context_manager.add_conversation_turn(
            "user", 
            command.voice_text,
            entities=[det.class_name for det in command.visual_detections]
        )
        
        # Resolve references using context manager
        resolved_text = self.context_manager.resolve_command_references(command.voice_text)
        
        # Enhance command with spatial references
        if command.spatial_references:
            enhanced_text = self._enhance_command_with_spatial_info(resolved_text, command.spatial_references)
            command.resolved_command = enhanced_text
        else:
            command.resolved_command = resolved_text
            
        # Calculate confidence score
        command.confidence_score = self._calculate_command_confidence(command)
        
        # Process with AI if available
        if self.ai_processor:
            try:
                # Prepare context for AI processor
                ai_context = {
                    'voice_command': command.voice_text,
                    'resolved_command': command.resolved_command,
                    'detected_objects': [det.class_name for det in command.visual_detections],
                    'spatial_references': command.spatial_references,
                    'confidence': command.confidence_score
                }
                
                response = await self.ai_processor.process_command_async(
                    command.resolved_command, 
                    context=ai_context
                )
                
                console.print(f"[green]AI Response: {response.get('action', 'No action')}[/green]")
                
                # Add AI response to context
                self.context_manager.add_conversation_turn(
                    "assistant",
                    response.get('response', ''),
                    intent=response.get('intent')
                )
                
            except Exception as e:
                logger.error(f"AI processing error: {e}")
                console.print(f"[red]AI processing failed: {e}[/red]")
                
        # Store in command history
        self.command_history.append(command)
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]
            
        # Show processing result
        if command.resolved_command != command.voice_text:
            console.print(f"[blue]Resolved: '{command.resolved_command}'[/blue]")
            
        if command.spatial_references:
            console.print(f"[dim]Spatial refs: {command.spatial_references}[/dim]")
            
    def _enhance_command_with_spatial_info(self, command: str, spatial_refs: Dict[str, Any]) -> str:
        """Enhance command text with resolved spatial information."""
        
        enhanced = command
        
        # Replace spatial references with specific object names
        replacements = {
            "that": spatial_refs.get("that"),
            "this": spatial_refs.get("this"),
            "the left object": spatial_refs.get("left_object"),
            "the right object": spatial_refs.get("right_object"),
            "the center object": spatial_refs.get("center_object"),
            "the large object": spatial_refs.get("large_object"),
            "the small object": spatial_refs.get("small_object"),
        }
        
        for placeholder, replacement in replacements.items():
            if replacement and placeholder in enhanced.lower():
                # Find the placeholder in the original case
                import re
                pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
                enhanced = pattern.sub(replacement, enhanced)
                
        return enhanced
        
    def _calculate_command_confidence(self, command: MultimodalCommand) -> float:
        """Calculate overall confidence score for a multimodal command."""
        
        # Base confidence from voice recognition
        confidence = command.voice_confidence
        
        # Boost confidence if we have visual confirmations
        if command.visual_detections:
            visual_boost = min(0.2, len(command.visual_detections) * 0.05)
            confidence += visual_boost
            
        # Boost confidence if spatial references were resolved
        if command.spatial_references:
            spatial_boost = min(0.15, len(command.spatial_references) * 0.03)
            confidence += spatial_boost
            
        # Penalize if no objects detected but command mentions objects
        command_lower = command.voice_text.lower()
        object_mentions = any(word in command_lower for word in 
                            ["pick", "grab", "take", "move", "put", "place", "block", "cup", "bottle"])
        
        if object_mentions and not command.visual_detections:
            confidence -= 0.1
            
        return min(1.0, max(0.0, confidence))
        
    def get_current_visual_frame(self) -> Optional[np.ndarray]:
        """Get current visual frame with overlays."""
        
        if self.current_frame is None or not self.visual_feedback:
            return self.current_frame
            
        return self.visual_feedback.render_frame(self.current_frame)
        
    def _update_fps(self):
        """Update FPS calculation."""
        
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            
    def get_interface_stats(self) -> Dict[str, Any]:
        """Get multimodal interface statistics."""
        
        return {
            "is_running": self.is_running,
            "fps": self.fps,
            "commands_processed": len(self.command_history),
            "current_detections": len(self.current_detections),
            "voice_enabled": self.config.enable_voice,
            "vision_enabled": self.config.enable_vision,
            "context_summary": self.context_manager.get_context_summary(),
            "visual_context": self.context_manager.get_visual_context_summary() if hasattr(self.context_manager, 'get_visual_context_summary') else {}
        }


async def main():
    """Example usage of MultimodalInterface."""
    
    console.print("[blue]Saorse Multimodal Interface Demo[/blue]\n")
    
    # Create multimodal interface
    config = MultimodalConfig(
        enable_voice=True,
        enable_vision=True,
        enable_visual_feedback=True,
        detection_confidence=0.5,
        camera_resolution=(1280, 720)
    )
    
    interface = MultimodalInterface(config)
    
    try:
        # Run for demo duration
        console.print("[blue]Starting multimodal demo for 30 seconds...[/blue]")
        
        # Start interface
        demo_task = asyncio.create_task(interface.start())
        
        # Run demo
        await asyncio.sleep(30)
        
        # Stop interface
        await interface.stop()
        
        # Show stats
        stats = interface.get_interface_stats()
        console.print(f"\n[blue]Demo completed![/blue]")
        console.print(f"Commands processed: {stats['commands_processed']}")
        console.print(f"Average FPS: {stats['fps']:.1f}")
        console.print(f"Final detections: {stats['current_detections']}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
        await interface.stop()
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
        await interface.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())