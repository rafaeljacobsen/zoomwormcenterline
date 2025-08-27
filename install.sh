#!/bin/bash

# Worm Segmentation Installation Script
# Installs all dependencies and sets up the environment correctly

set -e  # Exit on any error

echo "🐛 Worm Segmentation Installation Script"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on WSL
check_wsl() {
    if grep -q microsoft /proc/version 2>/dev/null; then
        print_status "Detected WSL environment"
        WSL=true
    else
        print_status "Native Linux environment detected"
        WSL=false
    fi
}

# Check for NVIDIA GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        GPU_AVAILABLE=true
    else
        print_warning "No NVIDIA GPU detected - will use CPU"
        GPU_AVAILABLE=false
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Update package list
    sudo apt update
    
    # Install essential packages
    sudo apt install -y \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        curl \
        wget \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgoogle-perftools4
    
    print_success "System dependencies installed"
}

# Install Miniconda
install_conda() {
    if command -v conda &> /dev/null; then
        print_status "Conda already installed"
        return
    fi
    
    print_status "Installing Miniconda..."
    
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    
    # Add to PATH
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    
    print_success "Miniconda installed"
}

# Create conda environment
create_environment() {
    print_status "Creating conda environment 'zoomworm'..."
    
    # Source conda
    source $HOME/miniconda3/bin/activate
    
    # Remove existing environment if it exists
    conda remove -n zoomworm --all -y 2>/dev/null || true
    
    # Create new environment
    conda create -n zoomworm python=3.10 -y
    
    print_success "Conda environment created"
}

# Install PyTorch with appropriate CUDA support
install_pytorch() {
    print_status "Installing PyTorch..."
    
    # Activate environment
    source $HOME/miniconda3/bin/activate zoomworm
    
    if [ "$GPU_AVAILABLE" = true ]; then
        print_status "Installing PyTorch with CUDA support..."
        conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
        
        # Install additional CUDA tools if needed
        conda install cudatoolkit=12.1 -c conda-forge -y
    else
        print_status "Installing PyTorch CPU-only version..."
        conda install pytorch torchvision cpuonly -c pytorch -y
    fi
    
    print_success "PyTorch installed"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Activate environment
    source $HOME/miniconda3/bin/activate zoomworm
    
    # Install core ML packages
    pip install \
        pytorch-lightning \
        segmentation-models-pytorch \
        albumentations \
        opencv-python \
        scikit-learn \
        matplotlib \
        pillow \
        wandb \
        timm \
        tqdm \
        seaborn
    
    # Install MPI for distributed training
    conda install mpi4py -y
    
    print_success "Python dependencies installed"
}

# Set up CUDA environment for WSL
setup_cuda_wsl() {
    if [ "$WSL" = true ] && [ "$GPU_AVAILABLE" = true ]; then
        print_status "Setting up CUDA environment for WSL..."
        
        # Add CUDA paths to bashrc
        cat >> ~/.bashrc << 'EOF'

# CUDA Environment for WSL
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
EOF
        
        # Export for current session
        export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        export CUDA_VISIBLE_DEVICES=0
        
        print_success "CUDA environment configured for WSL"
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Activate environment
    source $HOME/miniconda3/bin/activate zoomworm
    
    # Test PyTorch
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    
    # Test CUDA if available
    if [ "$GPU_AVAILABLE" = true ]; then
        CUDA_TEST=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$CUDA_TEST" = "True" ]; then
            print_success "CUDA working correctly!"
            python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
        else
            print_warning "CUDA not working - will use CPU"
        fi
    fi
    
    # Test other packages
    python -c "import pytorch_lightning as pl; print(f'PyTorch Lightning: {pl.__version__}')"
    python -c "import segmentation_models_pytorch as smp; print('SMP: OK')"
    python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
    python -c "import albumentations as A; print('Albumentations: OK')"
    
    print_success "All packages working correctly!"
}

# Create example usage script
create_usage_script() {
    print_status "Creating usage examples..."
    
    cat > run_examples.sh << 'EOF'
#!/bin/bash

# Worm Segmentation Usage Examples
# Make sure to activate the conda environment first: conda activate zoomworm

echo "🐛 Worm Segmentation Usage Examples"
echo "=================================="

echo ""
echo "1. Explore your dataset:"
echo "python explore_data.py --visualize"

echo ""
echo "2. Train the model (quick test):"
echo "python main.py --epochs 10 --batch-size 4 --no-wandb"

echo ""
echo "3. Train the model (full training):"
echo "python main.py --epochs 100 --batch-size 8"

echo ""
echo "4. Run inference on images:"
echo "python main.py --mode inference \\"
echo "    --checkpoint checkpoints/best-model.ckpt \\"
echo "    --input-folder /path/to/images \\"
echo "    --output-folder results"

echo ""
echo "5. Process video (fastest):"
echo "python video_inference.py \\"
echo "    --video /path/to/video.mp4 \\"
echo "    --checkpoint checkpoints/best-model.ckpt \\"
echo "    --output video_results \\"
echo "    --batch-size 32"

echo ""
echo "6. Visualize data pipeline:"
echo "python main.py --mode visualize"

echo ""
echo "Remember to:"
echo "- Update data paths in config.py"
echo "- Activate conda environment: conda activate zoomworm"
echo "- Check GPU with: python -c 'import torch; print(torch.cuda.is_available())'"
EOF

    chmod +x run_examples.sh
    
    print_success "Usage examples created (run_examples.sh)"
}

# Main installation function
main() {
    print_status "Starting installation..."
    
    # Run installation steps
    check_wsl
    check_gpu
    install_system_deps
    install_conda
    create_environment
    install_pytorch
    install_python_deps
    setup_cuda_wsl
    test_installation
    create_usage_script
    
    echo ""
    print_success "🎉 Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Restart your terminal or run: source ~/.bashrc"
    echo "2. Activate environment: conda activate zoomworm"
    echo "3. Update data paths in config.py"
    echo "4. Run: python explore_data.py --visualize"
    echo ""
    echo "For usage examples, run: ./run_examples.sh"
}

# Handle interruption
trap 'print_error "Installation interrupted"; exit 1' INT

# Run main function
main "$@" 