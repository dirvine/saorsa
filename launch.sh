#!/bin/bash
# Saorse Launch Script

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}           Saorse Robot Control         ${NC}"
echo -e "${BLUE}     Voice-Controlled SO-101 Arms      ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if virtual environment exists
if [[ ! -d "venv" ]]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please run the installation script first:"
    echo "  ./scripts/install_mac.sh"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Check for required ports
if [[ $# -eq 0 ]]; then
    echo -e "${YELLOW}Usage: $0 <leader_port> [follower_port]${NC}"
    echo "Example: $0 /dev/tty.usbserial-FT1234 /dev/tty.usbserial-FT5678"
    echo ""
    echo "Available serial ports:"
    ls /dev/tty.usbserial-* 2>/dev/null || echo "No USB serial ports found"
    echo ""
    echo "Other options:"
    echo "  $0 --test-audio           # Test microphone and speech recognition"
    echo "  $0 --test-robot <port>    # Test robot connection"
    echo "  $0 --calibrate <port>     # Calibrate robot"
    echo "  $0 --status              # Show system status"
    echo "  $0 --help               # Show all options"
    exit 1
fi

# Handle special commands
if [[ "$1" == "--test-audio" ]]; then
    echo -e "${BLUE}Testing audio system...${NC}"
    python src/main_mac.py test-audio
    exit $?
elif [[ "$1" == "--test-robot" ]]; then
    if [[ -z "$2" ]]; then
        echo -e "${RED}Error: Robot port required for test${NC}"
        echo "Usage: $0 --test-robot <port>"
        exit 1
    fi
    echo -e "${BLUE}Testing robot connection on $2...${NC}"
    python src/main_mac.py test-robot --port "$2"
    exit $?
elif [[ "$1" == "--calibrate" ]]; then
    if [[ -z "$2" ]]; then
        echo -e "${RED}Error: Robot port required for calibration${NC}"
        echo "Usage: $0 --calibrate <port>"
        exit 1
    fi
    echo -e "${BLUE}Starting robot calibration on $2...${NC}"
    python scripts/calibrate_robot.py "$2"
    exit $?
elif [[ "$1" == "--status" ]]; then
    echo -e "${BLUE}Checking system status...${NC}"
    python src/main_mac.py status
    exit $?
elif [[ "$1" == "--help" ]]; then
    python src/main_mac.py --help
    exit $?
fi

# Validate ports exist
if [[ ! -e "$1" ]]; then
    echo -e "${RED}Error: Leader port $1 not found${NC}"
    echo "Available ports:"
    ls /dev/tty.usbserial-* 2>/dev/null || echo "No USB serial ports found"
    exit 1
fi

if [[ -n "$2" && ! -e "$2" ]]; then
    echo -e "${RED}Error: Follower port $2 not found${NC}"
    echo "Available ports:"
    ls /dev/tty.usbserial-* 2>/dev/null || echo "No USB serial ports found"
    exit 1
fi

# Check if models are downloaded
if [[ ! -f "models/.gitkeep" ]] || [[ -z "$(ls -A models/ 2>/dev/null)" ]]; then
    echo -e "${YELLOW}Warning: AI models not found${NC}"
    echo "Downloading required models..."
    python scripts/download_models.py --all
    
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Error: Failed to download models${NC}"
        echo "Please run manually: python scripts/download_models.py"
        exit 1
    fi
fi

# Check configuration
CONFIG_FILE="configs/active.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${YELLOW}Creating active configuration...${NC}"
    
    # Use M3 config for Apple Silicon, default for others
    if [[ $(uname -m) == "arm64" ]]; then
        cp configs/mac_m3.yaml "$CONFIG_FILE"
        echo -e "${GREEN}Using M3 optimized configuration${NC}"
    else
        cp configs/default.yaml "$CONFIG_FILE"
        echo -e "${GREEN}Using default configuration${NC}"
    fi
fi

# Show startup info
echo -e "${GREEN}Starting Saorse with:${NC}"
echo "  Leader port: $1"
if [[ -n "$2" ]]; then
    echo "  Follower port: $2"
else
    echo "  Follower port: None (leader only)"
fi
echo "  Configuration: $CONFIG_FILE"
echo ""

# Pre-flight checks
echo -e "${BLUE}Running pre-flight checks...${NC}"

# Check microphone permissions on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # This is a simple check - full permission checking would require more complex code
    echo "✓ macOS detected"
fi

# Check Python environment
python -c "import torch; print('✓ PyTorch available')" 2>/dev/null || {
    echo -e "${RED}✗ PyTorch not available${NC}"
    exit 1
}

# Check MPS availability (Apple Silicon)
if [[ $(uname -m) == "arm64" ]]; then
    python -c "
import torch
if torch.backends.mps.is_available():
    print('✓ Metal Performance Shaders available')
else:
    print('⚠ MPS not available, using CPU')
" 2>/dev/null
fi

echo ""
echo -e "${GREEN}🚀 Launching Saorse...${NC}"
echo ""
echo -e "${YELLOW}Voice Commands:${NC}"
echo "  'robot, move to home position'"
echo "  'robot, open gripper'"
echo "  'robot, move left'"
echo "  'robot, stop'"
echo ""
echo -e "${YELLOW}Control:${NC}"
echo "  Ctrl+C to stop"
echo ""

# Launch Saorse with error handling
set -e
trap 'echo -e "\n${YELLOW}Saorse stopped${NC}"; exit 0' INT

# Run the main application
python src/main_mac.py run --leader-port "$1" ${2:+--follower-port "$2"} --config "$CONFIG_FILE"