# 🚀 Installation Guide - Worm Segmentation

Complete installation guide for the worm segmentation deep learning pipeline.

## 📋 System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 18.04+), Windows 10/11, or macOS
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 10GB free space
- **Python**: 3.8-3.11

### Recommended for GPU Training
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **CUDA**: 11.8 or 12.1
- **RAM**: 16GB+

## 🐧 Linux/WSL Installation (Recommended)

### Option 1: Automatic Installation
```bash
# Download and run the installation script
wget https://raw.githubusercontent.com/your-repo/worm-segmentation/main/install.sh
chmod +x install.sh
./install.sh
```

### Option 2: Manual Installation

#### Step 1: Install System Dependencies
```bash
sudo apt update
sudo apt install -y python3 python3-pip build-essential curl wget git
```

#### Step 2: Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### Step 3: Create Environment
```bash
conda create -n zoomworm python=3.10 -y
conda activate zoomworm
```

#### Step 4: Install PyTorch
**With GPU (CUDA):**
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

**CPU Only:**
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

#### Step 5: Install Dependencies
```bash
pip install pytorch-lightning segmentation-models-pytorch albumentations opencv-python scikit-learn matplotlib pillow wandb timm tqdm seaborn
conda install mpi4py -y
```

#### Step 6: WSL CUDA Setup (if using WSL with GPU)
```bash
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
source ~/.bashrc
```

## 🪟 Windows Installation

### Option 1: Automatic Installation
1. Download `install.bat` from the repository
2. Right-click and "Run as Administrator"
3. Follow the prompts

### Option 2: Manual Installation

#### Step 1: Install Miniconda
1. Download from: https://docs.conda.io/en/latest/miniconda.html
2. Install with default settings
3. Open "Anaconda Prompt"

#### Step 2: Create Environment
```cmd
conda create -n zoomworm python=3.10 -y
conda activate zoomworm
```

#### Step 3: Install PyTorch
**With GPU:**
```cmd
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

**CPU Only:**
```cmd
conda install pytorch torchvision cpuonly -c pytorch
```

#### Step 4: Install Dependencies
```cmd
pip install pytorch-lightning segmentation-models-pytorch albumentations opencv-python scikit-learn matplotlib pillow wandb timm tqdm seaborn
```

## 🍎 macOS Installation

### Step 1: Install Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Python and Dependencies
```bash
brew install python git
```

### Step 3: Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh -b
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Step 4: Create Environment and Install Packages
```bash
conda create -n zoomworm python=3.10 -y
conda activate zoomworm
conda install pytorch torchvision cpuonly -c pytorch
pip install pytorch-lightning segmentation-models-pytorch albumentations opencv-python scikit-learn matplotlib pillow wandb timm tqdm seaborn
```

## ✅ Verify Installation

### Test Basic Installation
```bash
conda activate zoomworm
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorch_lightning as pl; print(f'Lightning: {pl.__version__}')"
python -c "import segmentation_models_pytorch as smp; print('SMP: OK')"
```

### Test GPU (if available)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Test the Pipeline
```bash
# Update data paths in config.py first
python explore_data.py --help
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. CUDA Not Detected (WSL)
```bash
# Check if WSL CUDA driver is installed
ls /usr/lib/wsl/lib/libcuda.so.1

# If missing, update Windows to latest version
# Ensure NVIDIA driver 470+ is installed on Windows
```

#### 2. Import Errors
```bash
# Reinstall with exact versions
pip install --upgrade --force-reinstall torch torchvision
```

#### 3. Memory Errors
```bash
# Reduce batch size in config.py
batch_size = 4  # or even 2
```

#### 4. Permission Errors (Linux)
```bash
# Fix conda permissions
sudo chown -R $USER:$USER $HOME/miniconda3
```

#### 5. OpenCV Display Issues (WSL)
```bash
# Install X11 forwarding
sudo apt install x11-apps
# Or use --no-visualize flag
```

### Performance Optimization

#### For Training Speed
- Use GPU if available
- Increase batch size (8, 16, 32)
- Use mixed precision (automatic)
- Reduce image size if needed

#### For Memory Efficiency
- Reduce batch size (4, 2)
- Use smaller models (efficientnet-b2)
- Enable gradient checkpointing

#### For Inference Speed
- Use video processing for multiple frames
- Increase batch size
- Disable post-processing if not needed

## 📁 Project Structure After Installation

```
worm-segmentation/
├── config.py              # Configuration settings
├── dataset.py              # Data loading
├── models.py               # Neural network models
├── losses.py               # Loss functions
├── trainer.py              # Training pipeline
├── inference.py            # Single image inference
├── video_inference.py      # Video processing
├── main.py                 # Main training script
├── explore_data.py         # Dataset analysis
├── requirements.txt        # Python dependencies
├── install.sh             # Linux installation
├── install.bat            # Windows installation
└── README.md              # Documentation
```

## 🚀 Quick Start After Installation

1. **Activate Environment**
   ```bash
   conda activate zoomworm
   ```

2. **Update Data Paths**
   Edit `config.py` and set your data paths

3. **Explore Dataset**
   ```bash
   python explore_data.py --visualize
   ```

4. **Start Training**
   ```bash
   python main.py --epochs 10 --batch-size 4 --no-wandb
   ```

## 💡 Tips for Success

- **Start Small**: Use `--epochs 10` for initial testing
- **Monitor GPU**: Use `nvidia-smi` to check usage
- **Save Checkpoints**: Models auto-save to `checkpoints/`
- **Use Wandb**: Remove `--no-wandb` for experiment tracking
- **Batch Size**: Reduce if getting out-of-memory errors

## 🆘 Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Verify system requirements
3. Test with smaller batch sizes
4. Check GPU drivers and CUDA installation
5. Create an issue with error details

---

**Ready to segment some worms? 🐛🚀** 