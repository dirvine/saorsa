#!/usr/bin/env python3
"""
Context Manager for Saorse Robot System

This module provides advanced context-aware processing including spatial reasoning,
temporal context, object tracking, and multi-turn conversation management.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque, defaultdict

from rich.console import Console

# Vision integration imports
try:
    from object_detector import Detection, ObjectDetector
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    logger.warning("Vision modules not available for context integration")

console = Console()
logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context information."""
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    OBJECT = "object"
    CONVERSATION = "conversation"
    ROBOT_STATE = "robot_state"
    TASK = "task"
    ENVIRONMENT = "environment"


class ReferenceType(Enum):
    """Types of pronoun/reference resolution."""
    IT = "it"
    THAT = "that"
    THIS = "this"
    THERE = "there"
    HERE = "here"
    THE_OBJECT = "the_object"
    LAST_MENTIONED = "last_mentioned"


@dataclass
class SpatialContext:
    """Spatial context and workspace state."""
    robot_position: Dict[str, float] = field(default_factory=dict)
    gripper_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_state: str = "unknown"  # open, closed, partially_open
    
    # Object tracking
    known_objects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_pickup_location: Optional[Tuple[float, float, float]] = None
    last_dropoff_location: Optional[Tuple[float, float, float]] = None
    
    # Workspace regions
    workspace_regions: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "left": {"bounds": (-300, -100, -300, 300, 0, 400), "description": "left side of workspace"},
        "right": {"bounds": (100, 300, -300, 300, 0, 400), "description": "right side of workspace"},
        "center": {"bounds": (-100, 100, -100, 100, 0, 400), "description": "center of workspace"},
        "front": {"bounds": (-300, 300, -300, -100, 0, 400), "description": "front of workspace"},
        "back": {"bounds": (-300, 300, 100, 300, 0, 400), "description": "back of workspace"},
        "table": {"bounds": (-200, 200, -200, 200, 0, 50), "description": "table surface"}
    })


@dataclass 
class TemporalContext:
    """Temporal context and event history."""
    session_start_time: float = field(default_factory=time.time)
    last_command_time: float = 0.0
    last_action_time: float = 0.0
    last_movement_time: float = 0.0
    
    # Event history
    action_history: deque = field(default_factory=lambda: deque(maxlen=50))
    command_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Timing patterns
    average_command_interval: float = 0.0
    total_commands: int = 0


@dataclass
class ObjectContext:
    """Object tracking and properties."""
    id: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    last_seen_time: float = field(default_factory=time.time)
    estimated_position: Optional[Tuple[float, float, float]] = None
    confidence: float = 1.0
    
    # Object relationships
    related_objects: List[str] = field(default_factory=list)
    is_held: bool = False
    holder: Optional[str] = None  # gripper name or "human"
    
    # Vision-based tracking
    visual_detections: List['Detection'] = field(default_factory=list)
    pixel_position: Optional[Tuple[int, int]] = None  # Last known pixel coordinates
    visual_confidence: float = 0.0
    track_id: Optional[int] = None


@dataclass
class ConversationTurn:
    """Individual conversation turn."""
    timestamp: float
    speaker: str  # "user" or "assistant"
    text: str
    intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    resolved_references: Dict[str, str] = field(default_factory=dict)


@dataclass
class TaskContext:
    """High-level task context."""
    current_task: Optional[str] = None
    task_steps: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    task_start_time: Optional[float] = None
    task_goal: Optional[str] = None
    
    # Task patterns
    known_tasks: Dict[str, List[str]] = field(default_factory=lambda: {
        "pick_and_place": [
            "approach object",
            "open gripper", 
            "grasp object",
            "lift object",
            "move to target",
            "place object",
            "release object"
        ],
        "stack_blocks": [
            "identify blocks",
            "pick up first block",
            "place on base location",
            "pick up second block",
            "stack on first block",
            "verify stable stack"
        ],
        "organize_objects": [
            "identify objects to organize",
            "determine organization criteria",
            "pick up objects one by one",
            "place in organized arrangement",
            "verify organization"
        ]
    })


class ReferenceResolver:
    """Resolves pronouns and spatial references."""
    
    def __init__(self, context_manager):
        self.context_manager = context_manager
        
    def resolve_reference(self, text: str, reference_type: ReferenceType) -> Optional[str]:
        """Resolve a reference to a specific object or location."""
        
        if reference_type == ReferenceType.IT:
            return self._resolve_it_reference()
        elif reference_type == ReferenceType.THAT:
            return self._resolve_that_reference()
        elif reference_type == ReferenceType.THIS:
            return self._resolve_this_reference()
        elif reference_type == ReferenceType.THERE:
            return self._resolve_there_reference(text)
        elif reference_type == ReferenceType.HERE:
            return self._resolve_here_reference()
        elif reference_type == ReferenceType.THE_OBJECT:
            return self._resolve_the_object_reference()
        elif reference_type == ReferenceType.LAST_MENTIONED:
            return self._resolve_last_mentioned_reference()
            
        return None
        
    def _resolve_it_reference(self) -> Optional[str]:
        """Resolve 'it' reference to most recently mentioned object."""
        recent_turns = list(self.context_manager.conversation_context.turns)[-5:]
        
        for turn in reversed(recent_turns):
            for entity in turn.entities:
                if self._is_object_entity(entity):
                    return entity
                    
        # Check currently held object
        for obj in self.context_manager.object_context.values():
            if obj.is_held:
                return obj.name
                
        return None
        
    def _resolve_that_reference(self) -> Optional[str]:
        """Resolve 'that' reference (usually distant object)."""
        # Similar to 'it' but may prefer objects not currently held
        recent_objects = self._get_recent_objects()
        
        for obj_name in recent_objects:
            obj = self.context_manager.object_context.get(obj_name)
            if obj and not obj.is_held:
                return obj_name
                
        return self._resolve_it_reference()
        
    def _resolve_this_reference(self) -> Optional[str]:
        """Resolve 'this' reference (usually nearby object)."""
        # Prefer currently held object or very recently mentioned
        for obj in self.context_manager.object_context.values():
            if obj.is_held and obj.holder == "gripper":
                return obj.name
                
        return self._resolve_it_reference()
        
    def _resolve_there_reference(self, text: str) -> Optional[str]:
        """Resolve 'there' reference to a location."""
        # Look for spatial indicators in the text
        spatial_words = {
            "left": "left",
            "right": "right", 
            "front": "front",
            "back": "back",
            "center": "center",
            "table": "table"
        }
        
        text_lower = text.lower()
        for word, location in spatial_words.items():
            if word in text_lower:
                return location
                
        # Use last dropoff location if available
        if self.context_manager.spatial_context.last_dropoff_location:
            return "last_dropoff_location"
            
        return "center"  # Default fallback
        
    def _resolve_here_reference(self) -> Optional[str]:
        """Resolve 'here' reference to current gripper location."""
        return "current_position"
        
    def _resolve_the_object_reference(self) -> Optional[str]:
        """Resolve 'the object' reference."""
        # Look for most salient object in recent context
        recent_objects = self._get_recent_objects()
        
        if recent_objects:
            return recent_objects[0]
            
        return None
        
    def _resolve_last_mentioned_reference(self) -> Optional[str]:
        """Resolve to last explicitly mentioned object."""
        recent_turns = list(self.context_manager.conversation_context.turns)[-10:]
        
        for turn in reversed(recent_turns):
            if turn.speaker == "user":
                for entity in turn.entities:
                    if self._is_object_entity(entity):
                        return entity
                        
        return None
        
    def _get_recent_objects(self) -> List[str]:
        """Get recently mentioned objects in order of recency."""
        recent_objects = []
        recent_turns = list(self.context_manager.conversation_context.turns)[-10:]
        
        for turn in reversed(recent_turns):
            for entity in turn.entities:
                if self._is_object_entity(entity) and entity not in recent_objects:
                    recent_objects.append(entity)
                    
        return recent_objects
        
    def _is_object_entity(self, entity: str) -> bool:
        """Check if entity represents a physical object."""
        object_words = {
            "block", "cube", "ball", "bottle", "cup", "tool", "box", "item",
            "red", "blue", "green", "yellow", "black", "white", "orange", "purple"
        }
        
        entity_lower = entity.lower()
        return any(word in entity_lower for word in object_words)


class ContextManager:
    """
    Advanced context management for natural language robot interaction.
    
    Maintains spatial, temporal, object, and conversation context to enable
    sophisticated reference resolution and context-aware command processing.
    Integrates with computer vision for visual object tracking and spatial reasoning.
    """
    
    def __init__(self):
        self.spatial_context = SpatialContext()
        self.temporal_context = TemporalContext()
        self.object_context: Dict[str, ObjectContext] = {}
        self.conversation_context = type('ConversationContext', (), {
            'turns': deque(maxlen=100),
            'current_topic': None,
            'conversation_state': 'idle'
        })()
        self.task_context = TaskContext()
        
        self.reference_resolver = ReferenceResolver(self)
        
        # Context update callbacks
        self.context_update_callbacks: List[callable] = []
        
        # Vision integration
        self.vision_enabled = VISION_AVAILABLE
        self.frame_history: deque = deque(maxlen=30)  # Keep recent frames for analysis
        self.detection_history: deque = deque(maxlen=100)  # Keep detection history
        
        # Camera calibration for 3D spatial reasoning
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.camera_to_robot_transform: Optional[np.ndarray] = None
        
    def update_robot_state(self, robot_state: Dict[str, Any]):
        """Update context with current robot state."""
        current_time = time.time()
        
        # Update spatial context
        if 'joint_positions' in robot_state:
            self.spatial_context.robot_position.update(robot_state['joint_positions'])
            
        if 'end_effector_position' in robot_state:
            self.spatial_context.gripper_position = robot_state['end_effector_position']
            
        if 'gripper_state' in robot_state:
            self.spatial_context.gripper_state = robot_state['gripper_state']
            
        # Update temporal context
        self.temporal_context.last_action_time = current_time
        
        # Record state change
        self.temporal_context.action_history.append({
            'timestamp': current_time,
            'type': 'robot_state_update',
            'data': robot_state
        })
        
        self._notify_context_update('robot_state', robot_state)
        
    def add_conversation_turn(self, speaker: str, text: str, intent: Optional[str] = None,
                            entities: Optional[List[str]] = None):
        """Add a conversation turn to the context."""
        current_time = time.time()
        
        # Update temporal context
        if speaker == "user":
            self.temporal_context.last_command_time = current_time
            self.temporal_context.total_commands += 1
            
            # Update average command interval
            if self.temporal_context.total_commands > 1:
                time_since_start = current_time - self.temporal_context.session_start_time
                self.temporal_context.average_command_interval = time_since_start / self.temporal_context.total_commands
                
        # Create conversation turn
        turn = ConversationTurn(
            timestamp=current_time,
            speaker=speaker,
            text=text,
            intent=intent,
            entities=entities or []
        )
        
        # Resolve references in the text
        turn.resolved_references = self._resolve_references_in_text(text)
        
        # Add to conversation history
        self.conversation_context.turns.append(turn)
        self.temporal_context.command_history.append({
            'timestamp': current_time,
            'speaker': speaker,
            'text': text,
            'intent': intent
        })
        
        # Update conversation state
        self._update_conversation_state(text, intent)
        
        # Extract and update object context
        self._extract_and_update_objects(turn)
        
        self._notify_context_update('conversation', turn)
        
    def track_object(self, object_name: str, properties: Optional[Dict[str, Any]] = None,
                    position: Optional[Tuple[float, float, float]] = None):
        """Track an object in the context."""
        
        if object_name not in self.object_context:
            self.object_context[object_name] = ObjectContext(
                id=f"obj_{len(self.object_context)}",
                name=object_name
            )
            
        obj = self.object_context[object_name]
        obj.last_seen_time = time.time()
        
        if properties:
            obj.properties.update(properties)
            
        if position:
            obj.estimated_position = position
            
        self._notify_context_update('object_tracking', {
            'object': object_name,
            'properties': properties,
            'position': position
        })
        
    def update_object_state(self, object_name: str, is_held: bool, holder: Optional[str] = None):
        """Update object holding state."""
        if object_name in self.object_context:
            obj = self.object_context[object_name]
            obj.is_held = is_held
            obj.holder = holder
            
            if is_held and holder == "gripper":
                # Update spatial context
                self.spatial_context.last_pickup_location = obj.estimated_position
            elif not is_held and obj.estimated_position:
                # Update dropoff location
                self.spatial_context.last_dropoff_location = obj.estimated_position
                
    def start_task(self, task_name: str, goal: Optional[str] = None):
        """Start tracking a high-level task."""
        self.task_context.current_task = task_name
        self.task_context.task_goal = goal
        self.task_context.task_start_time = time.time()
        self.task_context.completed_steps = []
        
        # Set task steps if known
        if task_name in self.task_context.known_tasks:
            self.task_context.task_steps = self.task_context.known_tasks[task_name].copy()
        else:
            self.task_context.task_steps = []
            
        self._notify_context_update('task_start', {
            'task': task_name,
            'goal': goal
        })
        
    def complete_task_step(self, step_description: str):
        """Mark a task step as completed."""
        if self.task_context.current_task:
            self.task_context.completed_steps.append({
                'step': step_description,
                'timestamp': time.time()
            })
            
            # Remove from pending steps if it exists
            if step_description in self.task_context.task_steps:
                self.task_context.task_steps.remove(step_description)
                
    def get_spatial_reference(self, reference: str) -> Optional[Tuple[float, float, float]]:
        """Get spatial coordinates for a reference."""
        
        if reference in self.spatial_context.workspace_regions:
            bounds = self.spatial_context.workspace_regions[reference]["bounds"]
            # Return center of region
            x = (bounds[0] + bounds[1]) / 2
            y = (bounds[2] + bounds[3]) / 2
            z = (bounds[4] + bounds[5]) / 2
            return (x, y, z)
            
        elif reference == "current_position":
            return self.spatial_context.gripper_position
            
        elif reference == "last_pickup_location":
            return self.spatial_context.last_pickup_location
            
        elif reference == "last_dropoff_location":
            return self.spatial_context.last_dropoff_location
            
        # Check if it's an object name
        if reference in self.object_context:
            return self.object_context[reference].estimated_position
            
        return None
        
    def resolve_command_references(self, command_text: str) -> str:
        """Resolve references in a command to specific objects/locations."""
        resolved_text = command_text
        
        # Find and resolve references
        references = self._find_references_in_text(command_text)
        
        for ref_text, ref_type in references.items():
            resolved = self.reference_resolver.resolve_reference(command_text, ref_type)
            if resolved:
                resolved_text = resolved_text.replace(ref_text, resolved)
                
        return resolved_text
        
    def get_context_summary(self) -> Dict[str, Any]:
        """Get comprehensive context summary."""
        current_time = time.time()
        
        return {
            'session_info': {
                'duration': current_time - self.temporal_context.session_start_time,
                'total_commands': self.temporal_context.total_commands,
                'avg_command_interval': self.temporal_context.average_command_interval
            },
            'spatial': {
                'gripper_position': self.spatial_context.gripper_position,
                'gripper_state': self.spatial_context.gripper_state,
                'known_objects': len(self.object_context),
                'held_objects': [obj.name for obj in self.object_context.values() if obj.is_held]
            },
            'conversation': {
                'turns': len(self.conversation_context.turns),
                'current_topic': self.conversation_context.current_topic,
                'state': self.conversation_context.conversation_state
            },
            'task': {
                'current_task': self.task_context.current_task,
                'steps_completed': len(self.task_context.completed_steps),
                'steps_remaining': len(self.task_context.task_steps)
            }
        }
        
    def _resolve_references_in_text(self, text: str) -> Dict[str, str]:
        """Find and resolve all references in text."""
        references = self._find_references_in_text(text)
        resolved = {}
        
        for ref_text, ref_type in references.items():
            resolution = self.reference_resolver.resolve_reference(text, ref_type)
            if resolution:
                resolved[ref_text] = resolution
                
        return resolved
        
    def _find_references_in_text(self, text: str) -> Dict[str, ReferenceType]:
        """Find reference words in text."""
        text_lower = text.lower()
        references = {}
        
        reference_patterns = {
            r'\bit\b': ReferenceType.IT,
            r'\bthat\b': ReferenceType.THAT,
            r'\bthis\b': ReferenceType.THIS,
            r'\bthere\b': ReferenceType.THERE,
            r'\bhere\b': ReferenceType.HERE,
            r'\bthe object\b': ReferenceType.THE_OBJECT
        }
        
        import re
        for pattern, ref_type in reference_patterns.items():
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                references[match.group()] = ref_type
                
        return references
        
    def _update_conversation_state(self, text: str, intent: Optional[str]):
        """Update conversation state based on new input."""
        text_lower = text.lower()
        
        # Detect conversation state
        if any(word in text_lower for word in ["pick", "grab", "get", "take"]):
            self.conversation_context.conversation_state = "picking"
        elif any(word in text_lower for word in ["place", "put", "drop", "set"]):
            self.conversation_context.conversation_state = "placing"
        elif any(word in text_lower for word in ["move", "go", "turn"]):
            self.conversation_context.conversation_state = "moving"
        elif any(word in text_lower for word in ["stop", "halt", "emergency"]):
            self.conversation_context.conversation_state = "stopping"
        else:
            self.conversation_context.conversation_state = "idle"
            
    def _extract_and_update_objects(self, turn: ConversationTurn):
        """Extract object mentions and update object context."""
        
        # Enhanced object detection
        object_patterns = {
            r'\b(?:red|blue|green|yellow|black|white|orange|purple)\s+(?:block|cube|ball|bottle|cup)\b',
            r'\b(?:block|cube|ball|bottle|cup|tool|box|item)\b',
            r'\b(?:the|a|an)\s+(?:red|blue|green|yellow|black|white|orange|purple)\s+(?:one|thing|object)\b'
        }
        
        import re
        text_lower = turn.text.lower()
        
        for pattern in object_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                object_name = match.group().strip()
                
                # Clean up object name
                object_name = re.sub(r'^(?:the|a|an)\s+', '', object_name)
                
                if object_name not in turn.entities:
                    turn.entities.append(object_name)
                    
                # Track object
                self.track_object(object_name)
                
    def update_visual_detections(self, detections: List['Detection'], frame: Optional[np.ndarray] = None):
        """Update object context with visual detections."""
        if not self.vision_enabled:
            return
            
        current_time = time.time()
        
        # Store frame for analysis
        if frame is not None:
            self.frame_history.append({
                'timestamp': current_time,
                'frame': frame,
                'detections': detections
            })
            
        # Store detection history
        self.detection_history.append({
            'timestamp': current_time,
            'detections': detections
        })
        
        # Update object context with visual information
        for detection in detections:
            object_name = detection.class_name
            
            # Create or update object context
            if object_name not in self.object_context:
                self.object_context[object_name] = ObjectContext(
                    id=f"obj_{len(self.object_context)}",
                    name=object_name
                )
                
            obj = self.object_context[object_name]
            obj.last_seen_time = current_time
            obj.visual_confidence = detection.confidence
            obj.pixel_position = detection.center
            obj.track_id = getattr(detection, 'track_id', None)
            
            # Add to visual detection history
            obj.visual_detections.append(detection)
            if len(obj.visual_detections) > 10:
                obj.visual_detections = obj.visual_detections[-10:]
                
            # Estimate 3D position if camera calibration is available
            if self.camera_matrix is not None:
                world_pos = self._pixel_to_world_coordinates(detection.center, detection.bbox)
                if world_pos:
                    obj.estimated_position = world_pos
                    
            # Update object properties based on visual information
            obj.properties.update({
                'bbox': detection.bbox,
                'area': detection.area,
                'visual_confidence': detection.confidence,
                'last_pixel_position': detection.center
            })
            
        self._analyze_spatial_relationships(detections)
        self._notify_context_update('visual_detections', detections)
        
    def _pixel_to_world_coordinates(self, pixel_pos: Tuple[int, int], 
                                   bbox: Tuple[int, int, int, int]) -> Optional[Tuple[float, float, float]]:
        """Convert pixel coordinates to 3D world coordinates."""
        if self.camera_matrix is None or self.camera_to_robot_transform is None:
            return None
            
        try:
            # This is a simplified conversion - real implementation would need
            # depth information or assumptions about object placement
            
            # Assume objects are on table surface (z = table_height)
            table_height = 0.0  # Height of table surface in robot coordinates
            
            # Convert pixel to normalized camera coordinates
            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            
            x_cam = (pixel_pos[0] - cx) / fx
            y_cam = (pixel_pos[1] - cy) / fy
            
            # Estimate depth based on object size (simplified)
            # This would be much more sophisticated in a real system
            bbox_area = bbox[2] * bbox[3]
            estimated_depth = max(0.3, 1.0 - (bbox_area / (640 * 480)) * 0.5)  # Rough estimate
            
            # 3D point in camera coordinates
            camera_point = np.array([x_cam * estimated_depth, y_cam * estimated_depth, estimated_depth, 1.0])
            
            # Transform to robot coordinates
            robot_point = self.camera_to_robot_transform @ camera_point
            
            return (robot_point[0], robot_point[1], robot_point[2])
            
        except Exception as e:
            logger.warning(f"Pixel to world conversion failed: {e}")
            return None
            
    def _analyze_spatial_relationships(self, detections: List['Detection']):
        """Analyze spatial relationships between detected objects."""
        if len(detections) < 2:
            return
            
        # Analyze relative positions
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i+1:], i+1):
                relationship = self._determine_spatial_relationship(det1, det2)
                
                # Update object context with relationships
                obj1_name = det1.class_name
                obj2_name = det2.class_name
                
                if obj1_name in self.object_context:
                    obj1 = self.object_context[obj1_name]
                    if obj2_name not in obj1.related_objects:
                        obj1.related_objects.append(obj2_name)
                        obj1.properties[f'relationship_to_{obj2_name}'] = relationship
                        
    def _determine_spatial_relationship(self, det1: 'Detection', det2: 'Detection') -> str:
        """Determine spatial relationship between two detections."""
        center1 = det1.center
        center2 = det2.center
        
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        
        # Determine primary relationship
        if abs(dx) > abs(dy):
            if dx > 0:
                return "right_of"
            else:
                return "left_of"
        else:
            if dy > 0:
                return "below"
            else:
                return "above"
                
    def find_objects_by_description(self, description: str) -> List[str]:
        """Find objects matching a natural language description."""
        description_lower = description.lower()
        matching_objects = []
        
        # Direct name matching
        for obj_name, obj in self.object_context.items():
            if obj_name.lower() in description_lower:
                matching_objects.append(obj_name)
                continue
                
            # Color and type matching
            for word in description_lower.split():
                if word in obj_name.lower():
                    matching_objects.append(obj_name)
                    break
                    
        # Spatial relationship matching
        if "left" in description_lower:
            # Find objects on the left side
            for obj_name, obj in self.object_context.items():
                if obj.pixel_position and obj.pixel_position[0] < 320:  # Left half of 640px image
                    if obj_name not in matching_objects:
                        matching_objects.append(obj_name)
                        
        if "right" in description_lower:
            # Find objects on the right side
            for obj_name, obj in self.object_context.items():
                if obj.pixel_position and obj.pixel_position[0] > 320:  # Right half of 640px image
                    if obj_name not in matching_objects:
                        matching_objects.append(obj_name)
                        
        # Confidence filtering - prefer higher confidence detections
        if len(matching_objects) > 1:
            matching_objects.sort(key=lambda name: self.object_context[name].visual_confidence, reverse=True)
            
        return matching_objects
        
    def get_object_by_spatial_reference(self, reference: str) -> Optional[str]:
        """Get object name by spatial reference (e.g., 'the object on the left')."""
        reference_lower = reference.lower()
        
        if "left" in reference_lower:
            # Find leftmost object
            leftmost_obj = None
            leftmost_x = float('inf')
            
            for obj_name, obj in self.object_context.items():
                if obj.pixel_position and obj.pixel_position[0] < leftmost_x:
                    leftmost_x = obj.pixel_position[0]
                    leftmost_obj = obj_name
                    
            return leftmost_obj
            
        elif "right" in reference_lower:
            # Find rightmost object
            rightmost_obj = None
            rightmost_x = 0
            
            for obj_name, obj in self.object_context.items():
                if obj.pixel_position and obj.pixel_position[0] > rightmost_x:
                    rightmost_x = obj.pixel_position[0]
                    rightmost_obj = obj_name
                    
            return rightmost_obj
            
        elif "center" in reference_lower or "middle" in reference_lower:
            # Find object closest to center
            center_obj = None
            min_distance = float('inf')
            center_x, center_y = 320, 240  # Image center
            
            for obj_name, obj in self.object_context.items():
                if obj.pixel_position:
                    distance = np.sqrt((obj.pixel_position[0] - center_x)**2 + 
                                     (obj.pixel_position[1] - center_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        center_obj = obj_name
                        
            return center_obj
            
        elif "nearest" in reference_lower or "closest" in reference_lower:
            # Find object closest to robot (highest in image, typically)
            nearest_obj = None
            nearest_y = float('inf')
            
            for obj_name, obj in self.object_context.items():
                if obj.pixel_position and obj.pixel_position[1] < nearest_y:
                    nearest_y = obj.pixel_position[1]
                    nearest_obj = obj_name
                    
            return nearest_obj
            
        return None
        
    def set_camera_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                              camera_to_robot_transform: np.ndarray):
        """Set camera calibration parameters for 3D spatial reasoning."""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_to_robot_transform = camera_to_robot_transform
        
        console.print("[green]✓ Camera calibration set for spatial reasoning[/green]")
        
    def get_visual_context_summary(self) -> Dict[str, Any]:
        """Get summary of visual context information."""
        if not self.vision_enabled:
            return {"vision_enabled": False}
            
        current_time = time.time()
        
        # Recent detections
        recent_detections = []
        if self.detection_history:
            recent_detection_data = self.detection_history[-1]
            recent_detections = [det.class_name for det in recent_detection_data['detections']]
            
        # Objects with visual tracking
        visually_tracked_objects = {
            obj_name: {
                'confidence': obj.visual_confidence,
                'pixel_position': obj.pixel_position,
                'estimated_position': obj.estimated_position,
                'last_seen': current_time - obj.last_seen_time
            }
            for obj_name, obj in self.object_context.items()
            if obj.visual_detections
        }
        
        return {
            "vision_enabled": True,
            "recent_detections": recent_detections,
            "tracked_objects": len(visually_tracked_objects),
            "visually_tracked_objects": visually_tracked_objects,
            "frame_history_length": len(self.frame_history),
            "detection_history_length": len(self.detection_history),
            "camera_calibrated": self.camera_matrix is not None
        }

    def add_context_callback(self, callback: callable):
        """Add callback for context updates."""
        self.context_update_callbacks.append(callback)
        
    def _notify_context_update(self, context_type: str, data: Any):
        """Notify callbacks of context updates."""
        for callback in self.context_update_callbacks:
            try:
                callback(context_type, data)
            except Exception as e:
                logger.error(f"Context callback error: {e}")


async def main():
    """Example usage of Context Manager."""
    
    context_manager = ContextManager()
    
    # Add context update callback
    def context_callback(context_type: str, data: Any):
        console.print(f"[dim]Context update: {context_type}[/dim]")
        
    context_manager.add_context_callback(context_callback)
    
    console.print("[blue]Testing Context Manager...[/blue]\n")
    
    # Simulate conversation
    commands = [
        ("user", "pick up the red block"),
        ("assistant", "Moving to pick up the red block"),
        ("user", "put it on the table"),
        ("assistant", "Placing the red block on the table"),
        ("user", "now grab the blue cube"),
        ("assistant", "Moving to pick up the blue cube"),
        ("user", "stack it on top of that"),
        ("assistant", "Stacking the blue cube on the red block")
    ]
    
    for speaker, text in commands:
        if speaker == "user":
            # Extract entities (simplified)
            entities = []
            if "red block" in text:
                entities.append("red block")
            if "blue cube" in text:
                entities.append("blue cube")
                
            context_manager.add_conversation_turn(speaker, text, entities=entities)
            
            # Show reference resolution
            resolved = context_manager.resolve_command_references(text)
            if resolved != text:
                console.print(f"[cyan]Original: {text}[/cyan]")
                console.print(f"[green]Resolved: {resolved}[/green]")
            else:
                console.print(f"[cyan]Command: {text}[/cyan]")
        else:
            context_manager.add_conversation_turn(speaker, text)
            
        await asyncio.sleep(0.5)
        
    # Show context summary
    summary = context_manager.get_context_summary()
    console.print(f"\n[blue]Context Summary:[/blue]")
    console.print(json.dumps(summary, indent=2))
    
    # Test spatial references
    console.print(f"\n[blue]Spatial References:[/blue]")
    for ref in ["center", "left", "table", "current_position"]:
        pos = context_manager.get_spatial_reference(ref)
        console.print(f"  {ref}: {pos}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())