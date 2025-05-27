#!/bin/bash
# Saorse Installation Script for macOS
# This script installs all dependencies and sets up the Saorse robot control system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    log_error "This script is designed for macOS only"
    exit 1
fi

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    log_warning "This script is optimized for Apple Silicon (M1/M2/M3)"
    log_warning "Intel Macs may work but are not officially supported"
fi

log_info "Starting Saorse installation for macOS..."

# Check for required tools
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        log_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    else
        log_success "Homebrew found"
    fi
    
    # Check for Python 3.11+
    if ! command -v python3 &> /dev/null; then
        log_info "Installing Python 3.11..."
        brew install python@3.11
    else
        python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ $(echo "$python_version >= 3.11" | bc -l) -eq 0 ]]; then
            log_warning "Python version $python_version found, but 3.11+ recommended"
            log_info "Installing Python 3.11..."
            brew install python@3.11
        else
            log_success "Python $python_version found"
        fi
    fi
    
    # Check for Git
    if ! command -v git &> /dev/null; then
        log_info "Installing Git..."
        brew install git
    else
        log_success "Git found"
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    # Audio libraries
    brew install portaudio
    brew install ffmpeg
    
    # Development tools
    brew install cmake
    brew install pkg-config
    
    # Optional: Install device drivers for common USB-Serial adapters
    if [[ ! -d "/Library/Extensions/FTDIUSBSerialDriver.kext" ]]; then
        log_warning "FTDI USB Serial drivers not found"
        log_info "You may need to install FTDI drivers for USB-Serial communication"
        log_info "Download from: https://ftdichip.com/drivers/vcp-drivers/"
    fi
    
    log_success "System dependencies installed"
}

# Setup Python virtual environment
setup_python_env() {
    log_info "Setting up Python virtual environment..."
    
    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with Metal support for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        log_info "Installing PyTorch with Metal Performance Shaders support..."
        pip install torch torchvision torchaudio
    else
        log_info "Installing PyTorch (CPU version for Intel Mac)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    log_success "Python environment setup complete"
}

# Download AI models
download_models() {
    log_info "Downloading AI models..."
    
    # Create models directory
    mkdir -p models
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run model download script
    python scripts/download_models.py
    
    log_success "AI models downloaded"
}

# Setup configuration
setup_config() {
    log_info "Setting up configuration..."
    
    # Create logs directory
    mkdir -p logs
    
    # Copy appropriate config based on system
    if [[ $(uname -m) == "arm64" ]]; then
        cp configs/mac_m3.yaml configs/active.yaml
        log_success "M3 configuration activated"
    else
        cp configs/default.yaml configs/active.yaml
        log_success "Default configuration activated"
    fi
}

# Create launch script
create_launch_script() {
    log_info "Creating launch script..."
    
    cat > launch.sh << 'EOF'
#!/bin/bash
# Saorse Launch Script

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Check for required ports
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <leader_port> [follower_port]"
    echo "Example: $0 /dev/tty.usbserial-FT1234 /dev/tty.usbserial-FT5678"
    echo ""
    echo "Available serial ports:"
    ls /dev/tty.usbserial-* 2>/dev/null || echo "No USB serial ports found"
    exit 1
fi

# Launch Saorse
python src/main_mac.py run --leader-port "$1" ${2:+--follower-port "$2"}
EOF

    chmod +x launch.sh
    log_success "Launch script created"
}

# Test installation
test_installation() {
    log_info "Testing installation..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Test audio system
    log_info "Testing audio system..."
    python src/main_mac.py test-audio
    
    # Test PyTorch MPS (Apple Silicon only)
    if [[ $(uname -m) == "arm64" ]]; then
        log_info "Testing PyTorch Metal Performance Shaders..."
        python -c "
import torch
if torch.backends.mps.is_available():
    print('✓ MPS is available')
    device = torch.device('mps')
    x = torch.randn(1, 3, 224, 224, device=device)
    print('✓ MPS tensor operations working')
else:
    print('❌ MPS not available')
"
    fi
    
    log_success "Installation test complete"
}

# Display completion message
display_completion() {
    log_success "Saorse installation complete!"
    echo ""
    echo "Next steps:"
    echo "1. Connect your SO-101 robot arms via USB"
    echo "2. Find your serial ports: ls /dev/tty.usbserial-*"
    echo "3. Launch Saorse: ./launch.sh /dev/tty.usbserial-XXXXXX"
    echo ""
    echo "For more information, see:"
    echo "- docs/hardware_setup.md - Hardware connection guide"
    echo "- docs/software_setup.md - Software configuration"
    echo "- docs/voice_commands.md - Available voice commands"
    echo ""
    echo "Test commands:"
    echo "- ./launch.sh --help           # Show all options"
    echo "- python src/main_mac.py status # Check system status"
    echo "- python src/main_mac.py test-audio # Test microphone"
    echo ""
    log_info "Happy robot controlling! 🤖"
}

# Main installation flow
main() {
    echo "================================================================="
    echo "                 Saorse Installation Script"
    echo "           Voice-Controlled SO-101 Robot Arms"
    echo "================================================================="
    echo ""
    
    # Run installation steps
    check_requirements
    install_system_deps
    setup_python_env
    download_models
    setup_config
    create_launch_script
    test_installation
    display_completion
}

# Handle interruption
trap 'log_error "Installation interrupted"; exit 1' INT

# Run main installation
main