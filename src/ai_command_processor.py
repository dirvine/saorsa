#!/usr/bin/env python3
"""
AI Command Processor for Saorsa Robot System

This module provides advanced natural language command processing using
local Hugging Face models including Pi-Zero, SmolLM2, and Qwen2.5.
"""

import asyncio
import logging
import time
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModel,
        pipeline, Pipeline, TextStreamer
    )
    from accelerate import infer_auto_device_map
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError as e:
    logging.error(f"Required AI libraries not found: {e}")
    raise

console = Console()
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported AI model types."""
    LEROBOT_PI0 = "lerobot/pi0"
    SMOLLM2_135M = "HuggingFaceTB/SmolLM2-135M-Instruct"
    SMOLLM2_360M = "HuggingFaceTB/SmolLM2-360M-Instruct" 
    SMOLLM2_1_7B = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    QWEN2_5_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
    PHI3_MINI = "microsoft/Phi-3-mini-4k-instruct"


class CommandType(Enum):
    """Types of robot commands."""
    MOVEMENT = "movement"
    GRIPPER = "gripper"
    POSITION = "position"
    SEQUENCE = "sequence"
    QUERY = "query"
    SAFETY = "safety"
    UNKNOWN = "unknown"


@dataclass
class CommandIntent:
    """Parsed command intent from natural language."""
    command_type: CommandType
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    original_text: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for ongoing conversation."""
    history: List[Dict[str, str]] = field(default_factory=list)
    current_robot_state: Dict[str, Any] = field(default_factory=dict)
    last_objects_mentioned: List[str] = field(default_factory=list)
    last_positions: Dict[str, Any] = field(default_factory=dict)
    session_start_time: float = field(default_factory=time.time)


class LocalModelManager:
    """Manages local Hugging Face models for robot command processing."""
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Pipeline] = {}
        self.current_model: Optional[str] = None
        
        # Determine best device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                
        console.print(f"[blue]AI Model Manager initialized on device: {self.device}[/blue]")
        
    def load_model(self, model_type: ModelType, force_reload: bool = False) -> bool:
        """Load a specific model."""
        model_name = model_type.value
        
        if model_name in self.models and not force_reload:
            console.print(f"[yellow]Model {model_name} already loaded[/yellow]")
            return True
            
        try:
            console.print(f"[blue]Loading model: {model_name}[/blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Loading {model_name}...", total=None)
                
                # Special handling for different model types
                if model_type == ModelType.LEROBOT_PI0:
                    # Pi-Zero robotics model
                    self._load_lerobot_model(model_name)
                else:
                    # Standard language models
                    self._load_language_model(model_name)
                    
                progress.update(task, completed=True)
                
            self.current_model = model_name
            console.print(f"[green]✓ Model {model_name} loaded successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Failed to load model {model_name}: {e}[/red]")
            logger.error(f"Model loading error: {e}")
            return False
            
    def _load_language_model(self, model_name: str):
        """Load a standard language model."""
        # Load tokenizer
        self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Ensure pad token exists
        if self.tokenizers[model_name].pad_token is None:
            self.tokenizers[model_name].pad_token = self.tokenizers[model_name].eos_token
            
        # Load model with device optimization
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
        }
        
        # For MPS, we need to be careful with device placement
        if self.device == "mps":
            model_kwargs["device_map"] = None  # Let us handle device placement
        elif self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            
        self.models[model_name] = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device if needed
        if self.device == "mps":
            try:
                self.models[model_name] = self.models[model_name].to(self.device)
            except Exception as e:
                logger.warning(f"Could not move model to MPS, using CPU: {e}")
                self.device = "cpu"
                self.models[model_name] = self.models[model_name].to("cpu")
                
        # Create text generation pipeline
        self.pipelines[model_name] = pipeline(
            "text-generation",
            model=self.models[model_name],
            tokenizer=self.tokenizers[model_name],
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=256,
            pad_token_id=self.tokenizers[model_name].eos_token_id
        )
        
    def _load_lerobot_model(self, model_name: str):
        """Load LeRobot Pi-Zero model (placeholder for future implementation)."""
        # Note: This is a placeholder for Pi-Zero integration
        # The actual Pi-Zero model would require LeRobot framework
        console.print(f"[yellow]Pi-Zero model loading is a placeholder for future implementation[/yellow]")
        
        # For now, we'll use a smaller language model for command processing
        # and plan to integrate Pi-Zero for action execution in the future
        fallback_model = "HuggingFaceTB/SmolLM2-360M-Instruct"
        console.print(f"[blue]Using fallback model: {fallback_model}[/blue]")
        self._load_language_model(fallback_model)
        
    def generate_response(self, prompt: str, model_name: Optional[str] = None, 
                         max_length: int = 200) -> str:
        """Generate response using the specified model."""
        if model_name is None:
            model_name = self.current_model
            
        if model_name not in self.pipelines:
            logger.error(f"Model {model_name} not loaded")
            return ""
            
        try:
            # Generate response
            result = self.pipelines[model_name](
                prompt,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizers[model_name].eos_token_id
            )
            
            # Extract generated text
            generated_text = result[0]["generated_text"]
            
            # Remove the original prompt from the response
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
                
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            "current_device": self.device,
            "current_model": self.current_model,
            "loaded_models": list(self.models.keys()),
            "available_models": [model.value for model in ModelType]
        }
        
        # Add model sizes if available
        for model_name in self.models:
            try:
                num_params = sum(p.numel() for p in self.models[model_name].parameters())
                info[f"{model_name}_parameters"] = f"{num_params:,}"
            except Exception:
                pass
                
        return info


class AICommandProcessor:
    """
    Advanced AI-powered command processor for natural language robot control.
    
    Uses local Hugging Face models to interpret complex commands, maintain context,
    and generate appropriate robot actions.
    """
    
    def __init__(self, model_manager: LocalModelManager):
        self.model_manager = model_manager
        self.context = ConversationContext()
        
        # Command patterns for different types
        self.movement_patterns = [
            r"move|go|turn|rotate|shift",
            r"left|right|forward|back|up|down",
            r"slowly|quickly|fast|slow"
        ]
        
        self.gripper_patterns = [
            r"grip|grasp|grab|hold|catch|pick",
            r"release|drop|let go|open|close",
            r"squeeze|tight|loose"
        ]
        
        self.object_patterns = [
            r"block|cube|ball|bottle|cup|tool",
            r"red|blue|green|yellow|black|white",
            r"small|large|big|tiny|heavy|light"
        ]
        
        self.position_patterns = [
            r"home|ready|rest|center",
            r"position|pose|posture",
            r"here|there|over|under|above|below"
        ]
        
    def process_command(self, text: str, robot_state: Optional[Dict[str, Any]] = None) -> CommandIntent:
        """Process natural language command and return structured intent."""
        
        # Update context
        if robot_state:
            self.context.current_robot_state.update(robot_state)
            
        # Add to conversation history
        self.context.history.append({
            "timestamp": time.time(),
            "user": text,
            "type": "command"
        })
        
        # First, try pattern-based classification for speed
        quick_intent = self._quick_classify_command(text)
        
        # If quick classification is confident enough, use it
        if quick_intent.confidence > 0.8:
            return quick_intent
            
        # Otherwise, use AI model for complex interpretation
        ai_intent = self._ai_interpret_command(text)
        
        # Combine insights from both approaches
        final_intent = self._merge_interpretations(quick_intent, ai_intent, text)
        
        # Update context with results
        self._update_context(final_intent)
        
        return final_intent
        
    def _quick_classify_command(self, text: str) -> CommandIntent:
        """Quick pattern-based command classification."""
        text_lower = text.lower()
        
        # Safety commands (highest priority)
        safety_words = ["stop", "halt", "emergency", "freeze", "abort"]
        if any(word in text_lower for word in safety_words):
            return CommandIntent(
                command_type=CommandType.SAFETY,
                action="emergency_stop",
                confidence=0.95,
                reasoning="Safety keyword detected",
                original_text=text
            )
            
        # Movement commands
        movement_score = sum(1 for pattern in self.movement_patterns 
                           if re.search(pattern, text_lower))
        
        # Gripper commands  
        gripper_score = sum(1 for pattern in self.gripper_patterns
                          if re.search(pattern, text_lower))
        
        # Position commands
        position_score = sum(1 for pattern in self.position_patterns
                           if re.search(pattern, text_lower))
        
        # Determine most likely command type
        scores = {
            CommandType.MOVEMENT: movement_score,
            CommandType.GRIPPER: gripper_score,
            CommandType.POSITION: position_score
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            return CommandIntent(
                command_type=CommandType.UNKNOWN,
                action="unknown",
                confidence=0.0,
                reasoning="No patterns matched",
                original_text=text
            )
            
        best_type = max(scores, key=scores.get)
        confidence = min(max_score / 3.0, 0.9)  # Scale to reasonable confidence
        
        # Extract parameters based on type
        parameters = self._extract_parameters(text_lower, best_type)
        
        return CommandIntent(
            command_type=best_type,
            action=self._determine_action(text_lower, best_type),
            parameters=parameters,
            confidence=confidence,
            reasoning=f"Pattern matching: {max_score} patterns",
            original_text=text
        )
        
    def _ai_interpret_command(self, text: str) -> CommandIntent:
        """Use AI model to interpret complex commands."""
        if not self.model_manager.current_model:
            return CommandIntent(
                command_type=CommandType.UNKNOWN,
                action="no_model",
                confidence=0.0,
                reasoning="No AI model loaded",
                original_text=text
            )
            
        # Create structured prompt for command interpretation
        prompt = self._create_interpretation_prompt(text)
        
        try:
            # Generate AI response
            response = self.model_manager.generate_response(prompt, max_length=150)
            
            # Parse AI response into structured intent
            intent = self._parse_ai_response(response, text)
            
            return intent
            
        except Exception as e:
            logger.error(f"AI interpretation error: {e}")
            return CommandIntent(
                command_type=CommandType.UNKNOWN,
                action="ai_error",
                confidence=0.0,
                reasoning=f"AI processing failed: {e}",
                original_text=text
            )
            
    def _create_interpretation_prompt(self, text: str) -> str:
        """Create a structured prompt for AI command interpretation."""
        
        # Get recent context
        recent_commands = self.context.history[-3:] if self.context.history else []
        context_str = ""
        if recent_commands:
            context_str = "Recent commands:\n" + "\n".join([
                f"- {cmd.get('user', '')}" for cmd in recent_commands
            ]) + "\n\n"
            
        # Current robot state
        robot_state_str = ""
        if self.context.current_robot_state:
            robot_state_str = f"Robot state: {self.context.current_robot_state}\n\n"
            
        prompt = f"""You are a robot command interpreter. Parse the following command into structured format.

{context_str}{robot_state_str}Command: "{text}"

Analyze this command and respond with:
1. Command type (movement, gripper, position, sequence, query, safety, unknown)
2. Specific action to take
3. Parameters (if any)
4. Confidence (0.0-1.0)

Format your response as:
Type: [command_type]
Action: [specific_action]
Parameters: [key=value pairs or none]
Confidence: [0.0-1.0]
Reasoning: [brief explanation]

Response:"""

        return prompt
        
    def _parse_ai_response(self, response: str, original_text: str) -> CommandIntent:
        """Parse AI model response into CommandIntent."""
        try:
            lines = response.strip().split('\n')
            parsed = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    parsed[key.strip().lower()] = value.strip()
                    
            # Extract command type
            command_type_str = parsed.get('type', 'unknown').lower()
            command_type = CommandType.UNKNOWN
            
            for ct in CommandType:
                if ct.value in command_type_str:
                    command_type = ct
                    break
                    
            # Extract action
            action = parsed.get('action', 'unknown')
            
            # Extract parameters
            parameters = {}
            param_str = parsed.get('parameters', '')
            if param_str and param_str != 'none':
                # Simple parameter parsing (can be enhanced)
                param_pairs = param_str.split(',')
                for pair in param_pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        parameters[key.strip()] = value.strip()
                        
            # Extract confidence
            confidence = 0.5  # Default
            try:
                confidence = float(parsed.get('confidence', '0.5'))
            except ValueError:
                pass
                
            reasoning = parsed.get('reasoning', 'AI interpretation')
            
            return CommandIntent(
                command_type=command_type,
                action=action,
                parameters=parameters,
                confidence=confidence,
                reasoning=reasoning,
                original_text=original_text
            )
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return CommandIntent(
                command_type=CommandType.UNKNOWN,
                action="parse_error",
                confidence=0.0,
                reasoning=f"Failed to parse AI response: {e}",
                original_text=original_text
            )
            
    def _merge_interpretations(self, quick_intent: CommandIntent, 
                             ai_intent: CommandIntent, text: str) -> CommandIntent:
        """Merge insights from pattern matching and AI interpretation."""
        
        # If one has much higher confidence, use it
        if abs(quick_intent.confidence - ai_intent.confidence) > 0.3:
            return quick_intent if quick_intent.confidence > ai_intent.confidence else ai_intent
            
        # If both agree on command type, merge parameters
        if quick_intent.command_type == ai_intent.command_type:
            merged_params = {**quick_intent.parameters, **ai_intent.parameters}
            avg_confidence = (quick_intent.confidence + ai_intent.confidence) / 2
            
            return CommandIntent(
                command_type=quick_intent.command_type,
                action=ai_intent.action if ai_intent.confidence > quick_intent.confidence else quick_intent.action,
                parameters=merged_params,
                confidence=avg_confidence,
                reasoning=f"Merged: {quick_intent.reasoning} + {ai_intent.reasoning}",
                original_text=text
            )
            
        # If they disagree, use the one with higher confidence
        return quick_intent if quick_intent.confidence > ai_intent.confidence else ai_intent
        
    def _extract_parameters(self, text: str, command_type: CommandType) -> Dict[str, Any]:
        """Extract parameters based on command type."""
        parameters = {}
        
        if command_type == CommandType.MOVEMENT:
            # Extract direction
            if "left" in text:
                parameters["direction"] = "left"
            elif "right" in text:
                parameters["direction"] = "right"
            elif "forward" in text or "front" in text:
                parameters["direction"] = "forward"
            elif "back" in text or "backward" in text:
                parameters["direction"] = "back"
            elif "up" in text:
                parameters["direction"] = "up"
            elif "down" in text:
                parameters["direction"] = "down"
                
            # Extract speed
            if "slow" in text:
                parameters["speed"] = "slow"
            elif "fast" in text or "quick" in text:
                parameters["speed"] = "fast"
                
            # Extract distance/amount
            numbers = re.findall(r'\d+', text)
            if numbers:
                parameters["amount"] = int(numbers[0])
                
        elif command_type == CommandType.GRIPPER:
            if "open" in text:
                parameters["state"] = "open"
            elif "close" in text:
                parameters["state"] = "close"
            elif "grab" in text or "grasp" in text:
                parameters["state"] = "close"
            elif "release" in text or "drop" in text:
                parameters["state"] = "open"
                
        elif command_type == CommandType.POSITION:
            if "home" in text:
                parameters["target"] = "home"
            elif "ready" in text:
                parameters["target"] = "ready"
            elif "rest" in text:
                parameters["target"] = "rest"
                
        return parameters
        
    def _determine_action(self, text: str, command_type: CommandType) -> str:
        """Determine specific action based on command type and text."""
        
        if command_type == CommandType.MOVEMENT:
            if any(word in text for word in ["turn", "rotate"]):
                return "rotate"
            else:
                return "move"
                
        elif command_type == CommandType.GRIPPER:
            if any(word in text for word in ["open", "release", "drop"]):
                return "open_gripper"
            else:
                return "close_gripper"
                
        elif command_type == CommandType.POSITION:
            if "home" in text:
                return "go_home"
            elif "ready" in text:
                return "go_ready"
            else:
                return "move_to_position"
                
        return "unknown"
        
    def _update_context(self, intent: CommandIntent):
        """Update conversation context with new intent."""
        
        # Add to history
        self.context.history.append({
            "timestamp": time.time(),
            "assistant": f"Interpreted as: {intent.action}",
            "type": "interpretation",
            "confidence": intent.confidence
        })
        
        # Extract and remember objects mentioned
        objects = re.findall(r'\b(?:' + '|'.join([
            'block', 'cube', 'ball', 'bottle', 'cup', 'tool',
            'red', 'blue', 'green', 'yellow', 'black', 'white'
        ]) + r')\b', intent.original_text.lower())
        
        self.context.last_objects_mentioned = list(set(objects))
        
        # Keep history manageable
        if len(self.context.history) > 20:
            self.context.history = self.context.history[-15:]
            
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation context."""
        return {
            "session_duration": time.time() - self.context.session_start_time,
            "total_commands": len([cmd for cmd in self.context.history if cmd.get("type") == "command"]),
            "recent_objects": self.context.last_objects_mentioned,
            "robot_state": self.context.current_robot_state,
            "recent_history": self.context.history[-5:]
        }


async def main():
    """Example usage of AI Command Processor."""
    
    # Initialize model manager
    model_manager = LocalModelManager()
    
    # Load a small model for testing
    success = model_manager.load_model(ModelType.SMOLLM2_360M)
    if not success:
        console.print("[red]Failed to load model, using pattern matching only[/red]")
        
    # Initialize command processor
    processor = AICommandProcessor(model_manager)
    
    # Test commands
    test_commands = [
        "robot move left",
        "open the gripper",
        "go to home position", 
        "pick up the red block",
        "put it over there",
        "emergency stop"
    ]
    
    console.print("[blue]Testing AI Command Processor...[/blue]\n")
    
    for command in test_commands:
        console.print(f"[cyan]Command: '{command}'[/cyan]")
        
        intent = processor.process_command(command)
        
        console.print(f"  Type: {intent.command_type.value}")
        console.print(f"  Action: {intent.action}")
        console.print(f"  Parameters: {intent.parameters}")
        console.print(f"  Confidence: {intent.confidence:.2f}")
        console.print(f"  Reasoning: {intent.reasoning}")
        console.print()
        
    # Show model info
    info = model_manager.get_model_info()
    console.print(f"[blue]Model Info: {info}[/blue]")
    
    # Show context
    context = processor.get_context_summary()
    console.print(f"[blue]Context: {context}[/blue]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())