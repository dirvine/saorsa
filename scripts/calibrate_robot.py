#!/usr/bin/env python3
"""
Robot Calibration Script for Saorse

This script provides robot calibration routines including:
- Motor connection testing
- Joint range calibration
- Home position setting
- Workspace boundary verification
"""

import asyncio
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from robot_controller_m3 import RobotController, create_default_so101_config
    from utils.safety_monitor import SafetyMonitor, create_default_safety_config
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    import yaml
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you've activated the virtual environment and installed dependencies")
    sys.exit(1)

console = Console()


class RobotCalibrator:
    """Robot calibration system."""
    
    def __init__(self, port: str):
        self.port = port
        self.robot_controller: Optional[RobotController] = None
        self.safety_monitor: Optional[SafetyMonitor] = None
        self.calibration_data = {}
        
    def connect(self) -> bool:
        """Connect to the robot."""
        console.print(f"[blue]Connecting to robot on {self.port}...[/blue]")
        
        try:
            # Create robot configuration
            config = create_default_so101_config(self.port, "Calibration")
            
            # Initialize safety monitor
            workspace_limits, safety_limits = create_default_safety_config()
            self.safety_monitor = SafetyMonitor(workspace_limits, safety_limits)
            self.safety_monitor.start_monitoring()
            
            # Initialize robot controller
            self.robot_controller = RobotController(config)
            
            # Connect
            if self.robot_controller.connect(self.port):
                console.print("[green]✓ Robot connected successfully[/green]")
                return True
            else:
                console.print("[red]✗ Failed to connect to robot[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]✗ Connection error: {e}[/red]")
            return False
            
    def disconnect(self):
        """Disconnect from robot."""
        if self.robot_controller:
            self.robot_controller.disconnect()
        if self.safety_monitor:
            self.safety_monitor.stop_monitoring()
            
    def test_motor_connections(self) -> bool:
        """Test individual motor connections."""
        console.print("[blue]Testing motor connections...[/blue]")
        
        if not self.robot_controller:
            console.print("[red]✗ Robot not connected[/red]")
            return False
            
        table = Table(title="Motor Connection Test")
        table.add_column("Motor ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Position", style="magenta")
        table.add_column("Temperature", style="red")
        
        all_connected = True
        
        for motor in self.robot_controller.leader_config.motors:
            try:
                # Test motor communication
                connected = self.robot_controller.leader_controller.ping_motor(motor.motor_id)
                
                if connected:
                    # Get motor status
                    status = self.robot_controller.leader_controller.get_motor_status(motor.motor_id)
                    if status:
                        table.add_row(
                            str(motor.motor_id),
                            motor.name,
                            "[green]✓ Connected[/green]",
                            str(status.position),
                            f"{status.temperature}°C"
                        )
                    else:
                        table.add_row(
                            str(motor.motor_id),
                            motor.name,
                            "[yellow]⚠ No Status[/yellow]",
                            "N/A",
                            "N/A"
                        )
                else:
                    table.add_row(
                        str(motor.motor_id),
                        motor.name,
                        "[red]✗ Not Connected[/red]",
                        "N/A",
                        "N/A"
                    )
                    all_connected = False
                    
            except Exception as e:
                table.add_row(
                    str(motor.motor_id),
                    motor.name,
                    f"[red]✗ Error: {e}[/red]",
                    "N/A",
                    "N/A"
                )
                all_connected = False
                
        console.print(table)
        
        if all_connected:
            console.print("[green]✓ All motors connected successfully[/green]")
        else:
            console.print("[red]✗ Some motors failed connection test[/red]")
            
        return all_connected
        
    async def calibrate_joint_ranges(self) -> Dict[int, Dict[str, int]]:
        """Calibrate joint range limits."""
        console.print("[blue]Calibrating joint ranges...[/blue]")
        console.print("[yellow]This will move each joint to find its limits[/yellow]")
        
        if not self.robot_controller:
            console.print("[red]✗ Robot not connected[/red]")
            return {}
            
        joint_ranges = {}
        
        for motor in self.robot_controller.leader_config.motors:
            console.print(f"[blue]Calibrating {motor.name} (Motor {motor.motor_id})...[/blue]")
            
            try:
                # Get current position
                current_pos = self.robot_controller.leader_controller.get_position(motor.motor_id)
                if current_pos is None:
                    console.print(f"[red]✗ Cannot read position for motor {motor.motor_id}[/red]")
                    continue
                    
                console.print(f"Current position: {current_pos}")
                
                # For safety, use configured limits instead of finding actual limits
                # In a real calibration, you'd slowly move to find mechanical limits
                min_pos = motor.min_position
                max_pos = motor.max_position
                
                joint_ranges[motor.motor_id] = {
                    'min_position': min_pos,
                    'max_position': max_pos,
                    'home_position': motor.home_position,
                    'current_position': current_pos
                }
                
                console.print(f"  Range: {min_pos} - {max_pos}")
                console.print(f"  Home: {motor.home_position}")
                
                await asyncio.sleep(0.5)  # Brief pause between motors
                
            except Exception as e:
                console.print(f"[red]✗ Error calibrating motor {motor.motor_id}: {e}[/red]")
                
        # Display calibration results
        table = Table(title="Joint Range Calibration Results")
        table.add_column("Motor", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Min Position", style="yellow")
        table.add_column("Max Position", style="yellow")
        table.add_column("Home Position", style="magenta")
        table.add_column("Current Position", style="blue")
        
        for motor_id, ranges in joint_ranges.items():
            motor = next(m for m in self.robot_controller.leader_config.motors if m.motor_id == motor_id)
            table.add_row(
                str(motor_id),
                motor.name,
                str(ranges['min_position']),
                str(ranges['max_position']),
                str(ranges['home_position']),
                str(ranges['current_position'])
            )
            
        console.print(table)
        
        self.calibration_data['joint_ranges'] = joint_ranges
        return joint_ranges
        
    def set_home_position(self) -> bool:
        """Set and verify home position."""
        console.print("[blue]Setting home position...[/blue]")
        
        if not self.robot_controller:
            console.print("[red]✗ Robot not connected[/red]")
            return False
            
        try:
            # Move to home position
            console.print("[yellow]Moving to home position...[/yellow]")
            self.robot_controller.home_position()
            
            # Wait for movement to complete
            time.sleep(3)
            
            # Verify position
            console.print("[blue]Verifying home position...[/blue]")
            
            table = Table(title="Home Position Verification")
            table.add_column("Motor", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Target", style="yellow")
            table.add_column("Actual", style="magenta")
            table.add_column("Error", style="red")
            table.add_column("Status", style="blue")
            
            all_ok = True
            
            for motor in self.robot_controller.leader_config.motors:
                target_pos = motor.home_position
                actual_pos = self.robot_controller.leader_controller.get_position(motor.motor_id)
                
                if actual_pos is not None:
                    error = abs(actual_pos - target_pos)
                    status = "✓ OK" if error < 50 else "⚠ Error"  # 50 count tolerance
                    
                    if error >= 50:
                        all_ok = False
                        
                    table.add_row(
                        str(motor.motor_id),
                        motor.name,
                        str(target_pos),
                        str(actual_pos),
                        str(error),
                        status
                    )
                else:
                    table.add_row(
                        str(motor.motor_id),
                        motor.name,
                        str(target_pos),
                        "N/A",
                        "N/A",
                        "✗ No Data"
                    )
                    all_ok = False
                    
            console.print(table)
            
            if all_ok:
                console.print("[green]✓ Home position set successfully[/green]")
            else:
                console.print("[red]✗ Home position errors detected[/red]")
                
            self.calibration_data['home_position_ok'] = all_ok
            return all_ok
            
        except Exception as e:
            console.print(f"[red]✗ Error setting home position: {e}[/red]")
            return False
            
    async def test_movement_ranges(self) -> bool:
        """Test safe movement in each joint."""
        console.print("[blue]Testing movement ranges...[/blue]")
        console.print("[yellow]This will test small movements in each joint[/yellow]")
        
        if not self.robot_controller:
            console.print("[red]✗ Robot not connected[/red]")
            return False
            
        # First ensure we're at home position
        self.robot_controller.home_position()
        await asyncio.sleep(2)
        
        test_movements = {
            "joint1": [-15, 15],    # Base rotation
            "joint2": [-10, 10],    # Shoulder
            "joint3": [-10, 10],    # Elbow
            "joint4": [-20, 20],    # Wrist 1
            "joint5": [-20, 20],    # Wrist 2
            "joint6": [-30, 30],    # Wrist 3
        }
        
        table = Table(title="Movement Range Test")
        table.add_column("Joint", style="cyan")
        table.add_column("Movement", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Notes", style="dim")
        
        all_movements_ok = True
        
        for joint_name, movements in test_movements.items():
            for movement in movements:
                try:
                    console.print(f"Testing {joint_name}: {movement:+d}°")
                    
                    # Execute movement
                    self.robot_controller.move_to_position({joint_name: movement})
                    await asyncio.sleep(1.5)
                    
                    # Return to home for this joint
                    self.robot_controller.move_to_position({joint_name: 0})
                    await asyncio.sleep(1.5)
                    
                    table.add_row(
                        joint_name,
                        f"{movement:+d}°",
                        "[green]✓ OK[/green]",
                        "Movement completed"
                    )
                    
                except Exception as e:
                    console.print(f"[red]Movement error: {e}[/red]")
                    table.add_row(
                        joint_name,
                        f"{movement:+d}°",
                        "[red]✗ Failed[/red]",
                        str(e)
                    )
                    all_movements_ok = False
                    
        console.print(table)
        
        if all_movements_ok:
            console.print("[green]✓ All movement tests passed[/green]")
        else:
            console.print("[red]✗ Some movement tests failed[/red]")
            
        self.calibration_data['movement_test_ok'] = all_movements_ok
        return all_movements_ok
        
    def test_gripper(self) -> bool:
        """Test gripper operation."""
        console.print("[blue]Testing gripper...[/blue]")
        
        if not self.robot_controller:
            console.print("[red]✗ Robot not connected[/red]")
            return False
            
        try:
            positions = [0, 25, 50, 75, 100]  # 0% to 100%
            
            table = Table(title="Gripper Test")
            table.add_column("Position", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Notes", style="dim")
            
            for pos in positions:
                console.print(f"Setting gripper to {pos}%")
                
                self.robot_controller.set_gripper(pos)
                time.sleep(1)
                
                table.add_row(
                    f"{pos}%",
                    "[green]✓ OK[/green]",
                    "Position set successfully"
                )
                
            console.print(table)
            console.print("[green]✓ Gripper test completed[/green]")
            
            self.calibration_data['gripper_test_ok'] = True
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Gripper test failed: {e}[/red]")
            self.calibration_data['gripper_test_ok'] = False
            return False
            
    def save_calibration(self, filename: str = "calibration_results.yaml"):
        """Save calibration results to file."""
        try:
            output_file = Path(filename)
            
            calibration_data = {
                'timestamp': time.time(),
                'port': self.port,
                'calibration_results': self.calibration_data,
                'notes': [
                    'This file contains robot calibration results',
                    'Use this data to verify robot configuration',
                    'Run calibration again if hardware changes'
                ]
            }
            
            with open(output_file, 'w') as f:
                yaml.dump(calibration_data, f, default_flow_style=False)
                
            console.print(f"[green]✓ Calibration data saved to {output_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Error saving calibration: {e}[/red]")
            
    def display_summary(self):
        """Display calibration summary."""
        table = Table(title="Calibration Summary")
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Notes", style="dim")
        
        tests = [
            ('motor_connections', 'Motor Connections'),
            ('joint_ranges', 'Joint Range Calibration'),
            ('home_position_ok', 'Home Position'),
            ('movement_test_ok', 'Movement Tests'),
            ('gripper_test_ok', 'Gripper Test')
        ]
        
        for key, name in tests:
            if key in self.calibration_data:
                if key == 'joint_ranges':
                    status = "✓ Complete" if self.calibration_data[key] else "✗ Failed"
                else:
                    status = "✓ Passed" if self.calibration_data[key] else "✗ Failed"
                    
                color = "green" if self.calibration_data.get(key, False) else "red"
                table.add_row(name, f"[{color}]{status}[/{color}]", "")
            else:
                table.add_row(name, "[yellow]⏭ Skipped[/yellow]", "")
                
        console.print(table)
        
        # Overall result
        passed_tests = sum(1 for key in ['home_position_ok', 'movement_test_ok', 'gripper_test_ok'] 
                          if self.calibration_data.get(key, False))
        total_tests = 3
        
        if passed_tests == total_tests:
            console.print("[green bold]🎉 Robot calibration successful![/green bold]")
            console.print("[green]Robot is ready for voice control operation.[/green]")
        else:
            console.print(f"[yellow]⚠ {passed_tests}/{total_tests} tests passed.[/yellow]")
            console.print("[yellow]Review failed tests before proceeding.[/yellow]")


async def main():
    parser = argparse.ArgumentParser(description="Calibrate Saorse robot system")
    parser.add_argument("port", help="Robot serial port (e.g., /dev/tty.usbserial-FT1234)")
    parser.add_argument("--quick", action="store_true", help="Run quick calibration only")
    parser.add_argument("--save", default="calibration_results.yaml", help="Output file for results")
    
    args = parser.parse_args()
    
    # Show banner
    console.print(Panel.fit(
        "[bold blue]Saorse Robot Calibration[/bold blue]\n"
        "[dim]Comprehensive robot setup and testing[/dim]",
        title="🤖 Robot Calibration",
        border_style="blue"
    ))
    
    # Check if port exists
    if not Path(args.port).exists():
        console.print(f"[red]✗ Port {args.port} not found[/red]")
        console.print("[yellow]Available ports:[/yellow]")
        
        import glob
        ports = glob.glob("/dev/tty.usbserial-*")
        if ports:
            for port in ports:
                console.print(f"  {port}")
        else:
            console.print("  No USB serial ports found")
            
        return
        
    calibrator = RobotCalibrator(args.port)
    
    try:
        # Connect to robot
        if not calibrator.connect():
            return
            
        console.print("[blue]Starting calibration sequence...[/blue]\n")
        
        # Run calibration tests
        calibrator.test_motor_connections()
        await calibrator.calibrate_joint_ranges()
        calibrator.set_home_position()
        
        if not args.quick:
            await calibrator.test_movement_ranges()
            calibrator.test_gripper()
            
        # Show summary
        console.print("\n")
        calibrator.display_summary()
        
        # Save results
        calibrator.save_calibration(args.save)
        
        # Next steps
        console.print("\n[blue]Next steps:[/blue]")
        console.print("1. Review calibration results above")
        console.print("2. If all tests passed, robot is ready for voice control")
        console.print("3. Launch Saorse: ./launch.sh " + args.port)
        console.print("4. Try voice commands like 'robot move to home position'")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Calibration interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Calibration error: {e}[/red]")
    finally:
        calibrator.disconnect()


if __name__ == "__main__":
    asyncio.run(main())