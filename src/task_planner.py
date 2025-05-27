#!/usr/bin/env python3
"""
Task Planner for Saorse Robot System

This module provides advanced task planning, sequence generation, and execution
monitoring for complex multi-step robot operations.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from rich.console import Console
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class PrimitiveAction(Enum):
    """Basic robot primitive actions."""
    MOVE_TO_POSITION = "move_to_position"
    MOVE_RELATIVE = "move_relative"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"
    SET_GRIPPER = "set_gripper"
    HOME_POSITION = "home_position"
    READY_POSITION = "ready_position"
    WAIT = "wait"
    CHECK_POSITION = "check_position"
    VERIFY_GRASP = "verify_grasp"
    APPROACH_OBJECT = "approach_object"
    RETREAT = "retreat"


@dataclass
class TaskStep:
    """Individual task step with parameters and constraints."""
    id: str
    action: PrimitiveAction
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 2
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class Task:
    """High-level task composed of multiple steps."""
    id: str
    name: str
    description: str
    steps: List[TaskStep] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


class TaskTemplate:
    """Reusable task templates for common operations."""
    
    @staticmethod
    def pick_and_place(object_name: str, pickup_location: Tuple[float, float, float],
                      place_location: Tuple[float, float, float]) -> Task:
        """Create a pick and place task."""
        
        steps = [
            TaskStep(
                id="approach_pickup",
                action=PrimitiveAction.MOVE_TO_POSITION,
                parameters={
                    "position": (pickup_location[0], pickup_location[1], pickup_location[2] + 100),
                    "description": f"Approach {object_name} pickup location"
                },
                postconditions=[f"near_{object_name}"]
            ),
            TaskStep(
                id="open_gripper_pickup",
                action=PrimitiveAction.OPEN_GRIPPER,
                parameters={"value": 100},
                preconditions=[f"near_{object_name}"],
                postconditions=["gripper_open"]
            ),
            TaskStep(
                id="descend_to_object",
                action=PrimitiveAction.MOVE_TO_POSITION,
                parameters={
                    "position": pickup_location,
                    "speed": "slow"
                },
                preconditions=["gripper_open"],
                postconditions=[f"at_{object_name}"]
            ),
            TaskStep(
                id="grasp_object",
                action=PrimitiveAction.CLOSE_GRIPPER,
                parameters={"value": 30, "force": "medium"},
                preconditions=[f"at_{object_name}"],
                postconditions=[f"holding_{object_name}"]
            ),
            TaskStep(
                id="verify_grasp",
                action=PrimitiveAction.VERIFY_GRASP,
                parameters={"object": object_name},
                preconditions=[f"holding_{object_name}"],
                postconditions=[f"confirmed_holding_{object_name}"]
            ),
            TaskStep(
                id="lift_object",
                action=PrimitiveAction.MOVE_RELATIVE,
                parameters={"z": 100, "speed": "slow"},
                preconditions=[f"confirmed_holding_{object_name}"],
                postconditions=[f"lifted_{object_name}"]
            ),
            TaskStep(
                id="move_to_place",
                action=PrimitiveAction.MOVE_TO_POSITION,
                parameters={
                    "position": (place_location[0], place_location[1], place_location[2] + 100),
                    "description": f"Move {object_name} to place location"
                },
                preconditions=[f"lifted_{object_name}"],
                postconditions=[f"above_place_location"]
            ),
            TaskStep(
                id="descend_to_place",
                action=PrimitiveAction.MOVE_TO_POSITION,
                parameters={
                    "position": place_location,
                    "speed": "slow"
                },
                preconditions=[f"above_place_location"],
                postconditions=[f"at_place_location"]
            ),
            TaskStep(
                id="release_object",
                action=PrimitiveAction.OPEN_GRIPPER,
                parameters={"value": 100},
                preconditions=[f"at_place_location"],
                postconditions=[f"released_{object_name}"]
            ),
            TaskStep(
                id="retreat_from_place",
                action=PrimitiveAction.MOVE_RELATIVE,
                parameters={"z": 50},
                preconditions=[f"released_{object_name}"],
                postconditions=["task_complete"]
            )
        ]
        
        return Task(
            id=f"pick_place_{object_name}",
            name="Pick and Place",
            description=f"Pick up {object_name} and place it at target location",
            steps=steps,
            context={
                "object_name": object_name,
                "pickup_location": pickup_location,
                "place_location": place_location
            }
        )
    
    @staticmethod
    def stack_objects(bottom_object: str, top_object: str,
                     bottom_location: Tuple[float, float, float],
                     top_location: Tuple[float, float, float],
                     stack_location: Tuple[float, float, float]) -> Task:
        """Create a task to stack two objects."""
        
        # First pick and place the bottom object
        bottom_task = TaskTemplate.pick_and_place(bottom_object, bottom_location, stack_location)
        
        # Then stack the top object
        stack_height = 50  # Assume 50mm height for bottom object
        top_stack_location = (stack_location[0], stack_location[1], stack_location[2] + stack_height)
        
        top_task = TaskTemplate.pick_and_place(top_object, top_location, top_stack_location)
        
        # Combine steps
        all_steps = bottom_task.steps + top_task.steps
        
        # Update step IDs to avoid conflicts
        for i, step in enumerate(top_task.steps):
            step.id = f"top_{step.id}"
            
        # Add dependencies
        top_task.steps[0].preconditions.append("task_complete")  # Wait for bottom object
        
        return Task(
            id=f"stack_{bottom_object}_{top_object}",
            name="Stack Objects",
            description=f"Stack {top_object} on top of {bottom_object}",
            steps=all_steps,
            context={
                "bottom_object": bottom_object,
                "top_object": top_object,
                "bottom_location": bottom_location,
                "top_location": top_location,
                "stack_location": stack_location
            }
        )
        
    @staticmethod
    def organize_objects(objects: List[Dict[str, Any]], 
                        target_arrangement: str = "line") -> Task:
        """Create a task to organize multiple objects."""
        
        steps = []
        
        # Calculate target positions based on arrangement
        if target_arrangement == "line":
            spacing = 80  # 80mm between objects
            base_x = -len(objects) * spacing / 2
            
            for i, obj_info in enumerate(objects):
                target_pos = (base_x + i * spacing, 0, 50)
                
                # Create pick and place for this object
                obj_task = TaskTemplate.pick_and_place(
                    obj_info["name"],
                    obj_info["current_position"],
                    target_pos
                )
                
                # Add steps with unique IDs
                for step in obj_task.steps:
                    step.id = f"obj{i}_{step.id}"
                    if i > 0:  # Add dependency on previous object
                        if step == obj_task.steps[0]:  # First step of this object
                            step.preconditions.append(f"obj{i-1}_task_complete")
                            
                steps.extend(obj_task.steps)
                
        return Task(
            id="organize_objects",
            name="Organize Objects",
            description=f"Arrange {len(objects)} objects in {target_arrangement}",
            steps=steps,
            context={
                "objects": objects,
                "arrangement": target_arrangement
            }
        )


class TaskPlanner:
    """
    Advanced task planner for robot operations.
    
    Handles task decomposition, dependency management, execution monitoring,
    and failure recovery for complex robot tasks.
    """
    
    def __init__(self, robot_controller, context_manager):
        self.robot_controller = robot_controller
        self.context_manager = context_manager
        
        # Task management
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        # Execution state
        self.is_executing = False
        self.current_task: Optional[Task] = None
        self.current_step: Optional[TaskStep] = None
        
        # Planning parameters
        self.max_concurrent_tasks = 1
        self.step_execution_timeout = 30.0
        self.enable_replanning = True
        
        # Callbacks
        self.task_callbacks: List[callable] = []
        self.step_callbacks: List[callable] = []
        
    def create_task_from_command(self, command_intent, context: Dict[str, Any]) -> Optional[Task]:
        """Create a task from a natural language command intent."""
        
        command_text = command_intent.original_text.lower()
        
        # Pick and place commands
        if any(phrase in command_text for phrase in ["pick up", "grab", "get"]):
            return self._create_pickup_task(command_intent, context)
            
        elif any(phrase in command_text for phrase in ["put", "place", "drop"]):
            return self._create_place_task(command_intent, context)
            
        elif "stack" in command_text:
            return self._create_stack_task(command_intent, context)
            
        elif any(phrase in command_text for phrase in ["organize", "arrange", "sort"]):
            return self._create_organize_task(command_intent, context)
            
        elif any(phrase in command_text for phrase in ["move to", "go to"]):
            return self._create_movement_task(command_intent, context)
            
        else:
            # Create simple single-step task
            return self._create_simple_task(command_intent, context)
            
    def _create_pickup_task(self, command_intent, context: Dict[str, Any]) -> Task:
        """Create a pickup task from command intent."""
        
        # Extract object from parameters or context
        object_name = command_intent.parameters.get("object", "object")
        
        # Determine pickup location
        pickup_location = self._get_object_location(object_name, context)
        
        # Create task steps
        steps = [
            TaskStep(
                action=PrimitiveAction.APPROACH_OBJECT,
                parameters={"object": object_name, "location": pickup_location}
            ),
            TaskStep(
                action=PrimitiveAction.OPEN_GRIPPER,
                parameters={"value": 100}
            ),
            TaskStep(
                action=PrimitiveAction.MOVE_TO_POSITION,
                parameters={"position": pickup_location, "speed": "slow"}
            ),
            TaskStep(
                action=PrimitiveAction.CLOSE_GRIPPER,
                parameters={"value": 30}
            ),
            TaskStep(
                action=PrimitiveAction.VERIFY_GRASP,
                parameters={"object": object_name}
            ),
            TaskStep(
                action=PrimitiveAction.MOVE_RELATIVE,
                parameters={"z": 50}
            )
        ]
        
        return Task(
            name="Pickup Object",
            description=f"Pick up {object_name}",
            steps=steps,
            context={"object_name": object_name, "pickup_location": pickup_location}
        )
        
    def _create_place_task(self, command_intent, context: Dict[str, Any]) -> Task:
        """Create a place task from command intent."""
        
        # Determine place location from parameters
        place_reference = command_intent.parameters.get("location", "center")
        place_location = self.context_manager.get_spatial_reference(place_reference)
        
        if not place_location:
            place_location = (0, 0, 50)  # Default center position
            
        steps = [
            TaskStep(
                action=PrimitiveAction.MOVE_TO_POSITION,
                parameters={"position": (place_location[0], place_location[1], place_location[2] + 50)}
            ),
            TaskStep(
                action=PrimitiveAction.MOVE_TO_POSITION,
                parameters={"position": place_location, "speed": "slow"}
            ),
            TaskStep(
                action=PrimitiveAction.OPEN_GRIPPER,
                parameters={"value": 100}
            ),
            TaskStep(
                action=PrimitiveAction.MOVE_RELATIVE,
                parameters={"z": 30}
            )
        ]
        
        return Task(
            name="Place Object",
            description=f"Place object at {place_reference}",
            steps=steps,
            context={"place_location": place_location, "place_reference": place_reference}
        )
        
    def _create_stack_task(self, command_intent, context: Dict[str, Any]) -> Task:
        """Create a stacking task."""
        
        # This is a placeholder - real implementation would parse objects from command
        bottom_object = "red block"
        top_object = "blue cube"
        
        bottom_location = self._get_object_location(bottom_object, context)
        top_location = self._get_object_location(top_object, context)
        stack_location = (0, 0, 50)  # Center of workspace
        
        return TaskTemplate.stack_objects(
            bottom_object, top_object,
            bottom_location, top_location, stack_location
        )
        
    def _create_organize_task(self, command_intent, context: Dict[str, Any]) -> Task:
        """Create an organization task."""
        
        # Placeholder - real implementation would identify objects to organize
        objects = [
            {"name": "red block", "current_position": (100, 100, 50)},
            {"name": "blue cube", "current_position": (-100, 100, 50)},
            {"name": "green ball", "current_position": (0, -100, 50)}
        ]
        
        arrangement = "line"  # Could be parsed from command
        
        return TaskTemplate.organize_objects(objects, arrangement)
        
    def _create_movement_task(self, command_intent, context: Dict[str, Any]) -> Task:
        """Create a simple movement task."""
        
        target = command_intent.parameters.get("target", "ready")
        
        if target == "home":
            action = PrimitiveAction.HOME_POSITION
        elif target == "ready":
            action = PrimitiveAction.READY_POSITION
        else:
            action = PrimitiveAction.MOVE_TO_POSITION
            
        steps = [
            TaskStep(
                action=action,
                parameters=command_intent.parameters
            )
        ]
        
        return Task(
            name="Move Robot",
            description=f"Move to {target}",
            steps=steps,
            context={"target": target}
        )
        
    def _create_simple_task(self, command_intent, context: Dict[str, Any]) -> Task:
        """Create a simple single-step task."""
        
        # Map command intent to primitive action
        action_mapping = {
            "open_gripper": PrimitiveAction.OPEN_GRIPPER,
            "close_gripper": PrimitiveAction.CLOSE_GRIPPER,
            "move_home": PrimitiveAction.HOME_POSITION,
            "move_ready": PrimitiveAction.READY_POSITION,
            "move_relative": PrimitiveAction.MOVE_RELATIVE
        }
        
        action = action_mapping.get(command_intent.action, PrimitiveAction.MOVE_TO_POSITION)
        
        steps = [
            TaskStep(
                action=action,
                parameters=command_intent.parameters
            )
        ]
        
        return Task(
            name="Simple Command",
            description=f"Execute {command_intent.action}",
            steps=steps,
            context=command_intent.parameters
        )
        
    def add_task(self, task: Task, priority: int = 1) -> str:
        """Add a task to the execution queue."""
        
        task.priority = priority
        self.active_tasks[task.id] = task
        
        # Insert in priority order
        inserted = False
        for i, task_id in enumerate(self.task_queue):
            if self.active_tasks[task_id].priority < priority:
                self.task_queue.insert(i, task.id)
                inserted = True
                break
                
        if not inserted:
            self.task_queue.append(task.id)
            
        self._notify_task_update(task, "added")
        
        console.print(f"[green]Task added: {task.name} (ID: {task.id})[/green]")
        return task.id
        
    async def execute_tasks(self):
        """Execute tasks from the queue."""
        
        self.is_executing = True
        
        try:
            while self.task_queue and self.is_executing:
                task_id = self.task_queue.pop(0)
                
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                    await self._execute_task(task)
                    
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            console.print(f"[red]Task execution failed: {e}[/red]")
        finally:
            self.is_executing = False
            self.current_task = None
            self.current_step = None
            
    async def _execute_task(self, task: Task):
        """Execute a single task."""
        
        self.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        task.start_time = time.time()
        
        console.print(f"[blue]Executing task: {task.name}[/blue]")
        self._notify_task_update(task, "started")
        
        try:
            # Create progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                
                task_progress = progress.add_task(f"Task: {task.name}", total=len(task.steps))
                
                for i, step in enumerate(task.steps):
                    
                    # Check if step can be executed (preconditions)
                    if not self._check_preconditions(step, task):
                        step.status = TaskStatus.BLOCKED
                        logger.warning(f"Step {step.id} blocked by preconditions")
                        continue
                        
                    self.current_step = step
                    success = await self._execute_step(step, task)
                    
                    if success:
                        step.status = TaskStatus.COMPLETED
                        progress.update(task_progress, advance=1)
                        task.progress = (i + 1) / len(task.steps)
                    else:
                        # Handle step failure
                        if step.retry_count < step.max_retries:
                            step.retry_count += 1
                            logger.info(f"Retrying step {step.id} (attempt {step.retry_count})")
                            i -= 1  # Retry the same step
                            continue
                        else:
                            step.status = TaskStatus.FAILED
                            task.status = TaskStatus.FAILED
                            logger.error(f"Step {step.id} failed after {step.max_retries} retries")
                            break
                            
            # Check final task status
            if all(step.status == TaskStatus.COMPLETED for step in task.steps):
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
                console.print(f"[green]✓ Task completed: {task.name}[/green]")
                self.completed_tasks.append(task)
            else:
                task.status = TaskStatus.FAILED
                console.print(f"[red]✗ Task failed: {task.name}[/red]")
                self.failed_tasks.append(task)
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            logger.error(f"Task execution error: {e}")
            self.failed_tasks.append(task)
            
        finally:
            task.end_time = time.time()
            del self.active_tasks[task.id]
            self._notify_task_update(task, "completed")
            
    async def _execute_step(self, step: TaskStep, task: Task) -> bool:
        """Execute a single task step."""
        
        step.status = TaskStatus.IN_PROGRESS
        step.start_time = time.time()
        
        console.print(f"  [cyan]Step: {step.action.value}[/cyan]")
        self._notify_step_update(step, "started")
        
        try:
            # Execute the primitive action
            success = await self._execute_primitive_action(step.action, step.parameters)
            
            if success:
                # Verify postconditions
                success = self._verify_postconditions(step, task)
                
            step.end_time = time.time()
            return success
            
        except Exception as e:
            step.error_message = str(e)
            step.end_time = time.time()
            logger.error(f"Step execution error: {e}")
            return False
        finally:
            self._notify_step_update(step, "completed")
            
    async def _execute_primitive_action(self, action: PrimitiveAction, 
                                      parameters: Dict[str, Any]) -> bool:
        """Execute a primitive robot action."""
        
        try:
            if action == PrimitiveAction.MOVE_TO_POSITION:
                if "position" in parameters:
                    # Convert position to joint angles (simplified)
                    joint_positions = self._position_to_joints(parameters["position"])
                    self.robot_controller.move_to_position(joint_positions)
                    
            elif action == PrimitiveAction.MOVE_RELATIVE:
                self.robot_controller.move_to_position(parameters)
                
            elif action == PrimitiveAction.OPEN_GRIPPER:
                value = parameters.get("value", 100)
                self.robot_controller.set_gripper(value)
                
            elif action == PrimitiveAction.CLOSE_GRIPPER:
                value = parameters.get("value", 0)
                self.robot_controller.set_gripper(value)
                
            elif action == PrimitiveAction.SET_GRIPPER:
                value = parameters.get("value", 50)
                self.robot_controller.set_gripper(value)
                
            elif action == PrimitiveAction.HOME_POSITION:
                self.robot_controller.home_position()
                
            elif action == PrimitiveAction.READY_POSITION:
                self.robot_controller.ready_position()
                
            elif action == PrimitiveAction.WAIT:
                duration = parameters.get("duration", 1.0)
                await asyncio.sleep(duration)
                
            # Simplified - real implementation would handle all actions
            
            return True
            
        except Exception as e:
            logger.error(f"Primitive action {action.value} failed: {e}")
            return False
            
    def _position_to_joints(self, position: Tuple[float, float, float]) -> Dict[str, float]:
        """Convert Cartesian position to joint angles (simplified inverse kinematics)."""
        
        # This is a very simplified conversion
        # Real implementation would use proper inverse kinematics
        x, y, z = position
        
        joint1 = np.arctan2(y, x) * 180 / np.pi  # Base rotation
        joint2 = -30  # Shoulder
        joint3 = 60   # Elbow
        joint4 = 0    # Wrist 1
        joint5 = -30  # Wrist 2
        joint6 = 0    # Wrist 3
        
        return {
            "joint1": joint1,
            "joint2": joint2,
            "joint3": joint3,
            "joint4": joint4,
            "joint5": joint5,
            "joint6": joint6
        }
        
    def _check_preconditions(self, step: TaskStep, task: Task) -> bool:
        """Check if step preconditions are met."""
        
        # Simplified precondition checking
        # Real implementation would check robot state, context, etc.
        
        for condition in step.preconditions:
            if not self._evaluate_condition(condition, task):
                return False
                
        return True
        
    def _verify_postconditions(self, step: TaskStep, task: Task) -> bool:
        """Verify step postconditions were achieved."""
        
        for condition in step.postconditions:
            if not self._evaluate_condition(condition, task):
                logger.warning(f"Postcondition failed: {condition}")
                return False
                
        return True
        
    def _evaluate_condition(self, condition: str, task: Task) -> bool:
        """Evaluate a condition string."""
        
        # Simplified condition evaluation
        # Real implementation would check actual robot/world state
        
        if condition == "gripper_open":
            return True  # Would check actual gripper state
        elif condition == "gripper_closed":
            return True  # Would check actual gripper state
        elif condition.startswith("near_"):
            return True  # Would check position proximity
        elif condition.startswith("holding_"):
            return True  # Would check grasp state
            
        return True  # Default to true for demo
        
    def _get_object_location(self, object_name: str, 
                           context: Dict[str, Any]) -> Tuple[float, float, float]:
        """Get object location from context or estimate."""
        
        # Try to get from context manager
        if hasattr(self.context_manager, 'object_context'):
            if object_name in self.context_manager.object_context:
                obj = self.context_manager.object_context[object_name]
                if obj.estimated_position:
                    return obj.estimated_position
                    
        # Default locations for common objects (demo)
        default_locations = {
            "red block": (100, 100, 50),
            "blue cube": (-100, 100, 50),
            "green ball": (0, -100, 50),
            "yellow bottle": (150, 0, 50)
        }
        
        return default_locations.get(object_name, (0, 0, 50))
        
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a task."""
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
        elif task_id in [t.id for t in self.completed_tasks]:
            task = next(t for t in self.completed_tasks if t.id == task_id)
        elif task_id in [t.id for t in self.failed_tasks]:
            task = next(t for t in self.failed_tasks if t.id == task_id)
        else:
            return None
            
        return {
            "id": task.id,
            "name": task.name,
            "status": task.status.value,
            "progress": task.progress,
            "steps_total": len(task.steps),
            "steps_completed": sum(1 for s in task.steps if s.status == TaskStatus.COMPLETED),
            "start_time": task.start_time,
            "end_time": task.end_time
        }
        
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
                
            return True
            
        return False
        
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of task execution."""
        
        return {
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "is_executing": self.is_executing,
            "current_task": self.current_task.name if self.current_task else None,
            "current_step": self.current_step.action.value if self.current_step else None
        }
        
    def add_task_callback(self, callback: callable):
        """Add callback for task events."""
        self.task_callbacks.append(callback)
        
    def add_step_callback(self, callback: callable):
        """Add callback for step events."""
        self.step_callbacks.append(callback)
        
    def _notify_task_update(self, task: Task, event: str):
        """Notify callbacks of task updates."""
        for callback in self.task_callbacks:
            try:
                callback(task, event)
            except Exception as e:
                logger.error(f"Task callback error: {e}")
                
    def _notify_step_update(self, step: TaskStep, event: str):
        """Notify callbacks of step updates."""
        for callback in self.step_callbacks:
            try:
                callback(step, event)
            except Exception as e:
                logger.error(f"Step callback error: {e}")


async def main():
    """Example usage of Task Planner."""
    
    # Mock robot controller and context manager
    class MockRobotController:
        def move_to_position(self, positions):
            console.print(f"  [dim]Robot moving to: {positions}[/dim]")
            
        def set_gripper(self, value):
            console.print(f"  [dim]Setting gripper to: {value}[/dim]")
            
        def home_position(self):
            console.print(f"  [dim]Moving to home position[/dim]")
            
        def ready_position(self):
            console.print(f"  [dim]Moving to ready position[/dim]")
            
    class MockContextManager:
        def get_spatial_reference(self, ref):
            return (0, 0, 50)
            
    robot = MockRobotController()
    context = MockContextManager()
    
    planner = TaskPlanner(robot, context)
    
    # Add callbacks
    def task_callback(task, event):
        console.print(f"[dim]Task {event}: {task.name}[/dim]")
        
    def step_callback(step, event):
        console.print(f"[dim]Step {event}: {step.action.value}[/dim]")
        
    planner.add_task_callback(task_callback)
    planner.add_step_callback(step_callback)
    
    console.print("[blue]Testing Task Planner...[/blue]\n")
    
    # Create and execute a pick and place task
    task = TaskTemplate.pick_and_place(
        "red block",
        (100, 100, 50),
        (-100, -100, 50)
    )
    
    task_id = planner.add_task(task)
    
    # Execute tasks
    await planner.execute_tasks()
    
    # Show execution summary
    summary = planner.get_execution_summary()
    console.print(f"\n[blue]Execution Summary: {summary}[/blue]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import numpy as np
    asyncio.run(main())