# Voice Commands Reference

This comprehensive guide covers all voice commands supported by the Saorse robot control system across its three operational phases.

## Command Overview

The Saorse system supports voice commands at three levels of sophistication:

1. **Phase 1**: Basic pattern-matched commands for direct robot control
2. **Phase 2**: AI-enhanced natural language commands with context awareness
3. **Phase 3**: Multimodal commands combining voice with computer vision

## Phase 1: Basic Voice Commands

These commands work in all operational modes and provide direct, reliable robot control.

### Movement Commands

#### Linear Movement
```
"move left"
"move right"
"move forward"
"move back" / "move backward"
"move up"
"move down"
```

**Parameters**: 
- Default movement: 10-degree increments
- Speed: Configurable in settings (default: medium)

#### Rotational Movement
```
"turn left"
"turn right"
"rotate left"
"rotate right"
```

**Parameters**:
- Default rotation: 15-degree increments
- Affects end-effector orientation

#### Movement Modifiers
```
"move left slowly"
"move right quickly"
"move up fast"
"turn left slow"
```

**Speed Modifiers**:
- `slowly` / `slow`: 30% of normal speed
- `quickly` / `fast`: 200% of normal speed
- `very slowly`: 10% of normal speed
- `very fast`: 300% of normal speed

### Gripper Control

#### Basic Gripper Commands
```
"open gripper"
"close gripper"
"open"
"close"
"grab"
"release"
```

#### Gripper Positioning
```
"open gripper fully"
"close gripper gently"
"open gripper halfway"
"close gripper tight"
```

**Parameters**:
- `fully`: 100% open
- `halfway`: 50% open/closed
- `gently`: Reduced force
- `tight`: Maximum force (within safety limits)

### Position Commands

#### Predefined Positions
```
"home"
"home position"
"go home"
"ready"
"ready position"
"go to ready"
```

**Position Definitions**:
- **Home**: All joints at 0 degrees, gripper open
- **Ready**: Optimized position for starting tasks

#### Custom Positions (if configured)
```
"position one"
"position two"
"saved position alpha"
"go to position beta"
```

### Control Commands

#### Stop Commands
```
"stop"
"halt"
"pause"
"freeze"
```

**Behavior**: Immediately stops current movement, maintains position

#### Emergency Commands
```
"emergency"
"emergency stop"
"e-stop"
"abort"
```

**Behavior**: Immediate stop, disables motors, requires manual reset

#### Reset Commands
```
"reset"
"restart"
"initialize"
"calibrate"
```

**Behavior**: Returns robot to known state, may require recalibration

## Phase 2: AI-Enhanced Commands

These commands use natural language processing for more intuitive robot control.

### Object Manipulation Commands

#### Basic Object Commands
```
"pick up the red block"
"grab the blue cup"
"take the green bottle"
"lift the small box"
```

#### Placement Commands
```
"put it on the table"
"place it over there"
"set it down here"
"drop it in the box"
```

#### Movement Commands
```
"move it to the left"
"bring it closer"
"push it away"
"slide it to the right"
```

### Task-Level Commands

#### Stacking Operations
```
"stack the blocks"
"pile the objects"
"build a tower with the cubes"
"stack them by size"
"stack from largest to smallest"
```

#### Organization Commands
```
"organize the objects"
"sort by color"
"arrange by size"
"group similar items"
"line them up"
"make it neat"
```

#### Complex Tasks
```
"clean up the workspace"
"sort everything by type"
"arrange the tools properly"
"organize the parts by function"
```

### Context-Aware References

#### Pronoun Resolution
```
"pick up the red block"
"now move it to the left"      # "it" refers to red block
"put that over there"          # "that" refers to last object
"grab this one instead"        # "this" refers to nearby object
```

#### Spatial References
```
"the object on the left"
"the item in the center"
"the thing near the edge"
"the block next to the cup"
```

#### Temporal References
```
"the last thing I mentioned"
"what you just picked up"
"the previous object"
"the item from before"
```

## Phase 3: Multimodal Commands

These commands combine voice with computer vision for spatial understanding.

### Vision-Based Object Selection

#### Spatial Location References
```
"pick up the object on the left"
"grab the item on the right side"
"take the thing in the center"
"get the object in the corner"
"pick up the item nearest to me"
"grab the farthest object"
```

#### Size-Based Selection
```
"pick up the largest object"
"grab the smallest item"
"take the biggest block"
"get the tiny piece"
"select the medium-sized cup"
```

#### Color-Based Selection
```
"pick up the red object"
"grab the blue item on the left"
"take the green block in the center"
"get the yellow cup near the edge"
```

#### Shape-Based Selection
```
"pick up the round object"
"grab the square block"
"take the cylindrical item"
"get the bottle-shaped object"
```

### Relative Positioning Commands

#### Spatial Relationships
```
"move the cup next to the bottle"
"put the block between the two objects"
"place it behind the red item"
"set it in front of the blue object"
"position it to the left of that"
```

#### Distance Relationships
```
"move it closer to the red block"
"put it farther from the edge"
"place it near the center"
"set it at the same distance"
```

### Visual Confirmation Commands

#### Object Identification
```
"what do you see?"
"identify the objects"
"show me what's there"
"list the visible items"
"count the objects"
```

#### Object Description
```
"describe the red object"
"tell me about the largest item"
"what color is that block?"
"how many blue objects are there?"
```

#### Workspace Analysis
```
"scan the workspace"
"analyze the scene"
"find all the blocks"
"locate the tools"
"identify the containers"
```

## Command Syntax and Patterns

### Command Structure

#### Basic Pattern
```
[Action] [Object] [Location/Modifier]
```

Examples:
- "pick up" + "the red block" + "on the left"
- "move" + "it" + "to the center"
- "place" + "the cup" + "next to the bottle"

#### Advanced Pattern
```
[Modifier] [Action] [Object] [Spatial Relation] [Reference Object]
```

Examples:
- "carefully" + "move" + "the fragile cup" + "next to" + "the blue plate"
- "quickly" + "stack" + "all the blocks" + "by" + "size"

### Command Modifiers

#### Speed Modifiers
- `slowly`, `carefully`, `gently`
- `quickly`, `fast`, `rapidly`
- `very slowly`, `extremely carefully`
- `as fast as possible`

#### Precision Modifiers
- `precisely`, `exactly`, `accurately`
- `roughly`, `approximately`, `about`
- `perfectly aligned`, `straight`

#### Force Modifiers
- `gently`, `softly`, `lightly`
- `firmly`, `securely`, `tight`
- `with minimal force`, `with maximum grip`

## Voice Recognition Tips

### Optimal Speech Patterns

#### Clear Articulation
- Speak clearly and distinctly
- Avoid mumbling or rushing
- Pause briefly between words
- Use consistent volume

#### Natural Pacing
- Normal conversational speed
- Slight pause before important words
- Emphasize action words
- End with clear completion

### Environmental Optimization

#### Microphone Setup
- 12-18 inches from mouth
- Minimize background noise
- Reduce echo and reverberation
- Use directional microphone if possible

#### Speaking Environment
- Quiet room preferred
- Consistent ambient noise level
- Avoid sudden loud sounds
- Turn off fans/air conditioning if noisy

### Error Recovery

#### Command Repetition
```
"I didn't understand"    → Repeat the command
"Please repeat"          → Speak more clearly
"Say that again"         → Use simpler words
```

#### Command Clarification
```
"Did you mean X?"        → Confirm with "yes" or "no"
"Which object?"          → Be more specific
"Left or right?"         → Choose direction
```

#### Alternative Phrasing
```
Original: "relocate the red object"
Alternative: "move the red block"

Original: "position it adjacent to"
Alternative: "put it next to"
```

## Command Customization

### Adding Custom Commands

#### Configuration File
```yaml
# configs/voice_commands.yaml
custom_commands:
  "my special position":
    action: "move_to_position"
    parameters:
      joint1: 45
      joint2: -30
      joint3: 90
      
  "pickup sequence":
    action: "execute_sequence"
    steps:
      - "open gripper"
      - "move down"
      - "close gripper"
      - "move up"
```

#### Macro Commands
```yaml
macros:
  "initialize workspace":
    commands:
      - "home"
      - "open gripper"
      - "scan the workspace"
      
  "clean shutdown":
    commands:
      - "release any object"
      - "home"
      - "power down safely"
```

### Alias Configuration

#### Command Aliases
```yaml
aliases:
  "grab": "pick up"
  "drop": "release"
  "center": "middle"
  "put down": "place"
  "lift": "pick up"
```

#### Language Variations
```yaml
language_variants:
  british:
    "colour": "color"
    "grey": "gray"
  casual:
    "thingy": "object"
    "stuff": "items"
```

## Troubleshooting Voice Commands

### Common Issues

#### Command Not Recognized
**Possible Causes:**
- Unclear speech
- Background noise
- Microphone issues
- Unknown command syntax

**Solutions:**
- Speak more clearly
- Reduce background noise
- Check microphone settings
- Use simpler command structure

#### Wrong Action Executed
**Possible Causes:**
- Similar-sounding commands
- Ambiguous object references
- Context misunderstanding

**Solutions:**
- Use more specific language
- Include object colors/descriptions
- Establish clear context first

#### Slow Response Time
**Possible Causes:**
- Complex AI processing
- Large language models
- System resource constraints

**Solutions:**
- Use Phase 1 commands for speed
- Optimize system performance
- Reduce model complexity

### Performance Optimization

#### For Speed
- Use basic Phase 1 commands
- Reduce AI model size
- Optimize system resources
- Use wired microphone

#### For Accuracy
- Speak clearly and slowly
- Use specific object descriptions
- Establish context explicitly
- Confirm commands when uncertain

#### For Reliability
- Use simple command structure
- Avoid complex nested instructions
- Break complex tasks into steps
- Use visual confirmation when available

This voice commands reference provides comprehensive coverage of all supported speech patterns and usage guidelines for optimal robot control through natural language.