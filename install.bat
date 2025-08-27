@echo off
REM Worm Segmentation Installation Script for Windows
REM Installs all dependencies and sets up the environment correctly

echo 🐛 Worm Segmentation Installation Script for Windows
echo =====================================================

REM Check if conda is installed
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda not found. Please install Miniconda or Anaconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [INFO] Conda found, proceeding with installation...

REM Check for NVIDIA GPU
nvidia-smi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] NVIDIA GPU detected
    set GPU_AVAILABLE=true
) else (
    echo [WARNING] No NVIDIA GPU detected - will use CPU
    set GPU_AVAILABLE=false
)

REM Remove existing environment if it exists
echo [INFO] Removing existing zoomworm environment...
conda remove -n zoomworm --all -y >nul 2>nul

REM Create new environment
echo [INFO] Creating conda environment 'zoomworm'...
conda create -n zoomworm python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create conda environment
    pause
    exit /b 1
)

REM Activate environment
echo [INFO] Activating environment...
call conda activate zoomworm

REM Install PyTorch
if "%GPU_AVAILABLE%"=="true" (
    echo [INFO] Installing PyTorch with CUDA support...
    call conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
) else (
    echo [INFO] Installing PyTorch CPU-only version...
    call conda install pytorch torchvision cpuonly -c pytorch -y
)

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)

REM Install Python dependencies
echo [INFO] Installing Python dependencies...
call pip install pytorch-lightning segmentation-models-pytorch albumentations opencv-python scikit-learn matplotlib pillow wandb timm tqdm seaborn

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Python dependencies
    pause
    exit /b 1
)

REM Test installation
echo [INFO] Testing installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

if "%GPU_AVAILABLE%"=="true" (
    python -c "import torch; print('CUDA available:'); print(torch.cuda.is_available())"
)

python -c "import pytorch_lightning as pl; print(f'PyTorch Lightning: {pl.__version__}')"
python -c "import segmentation_models_pytorch as smp; print('SMP: OK')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

REM Create batch file for easy activation
echo @echo off > activate_zoomworm.bat
echo call conda activate zoomworm >> activate_zoomworm.bat
echo echo Environment activated! >> activate_zoomworm.bat
echo echo Run: python explore_data.py --visualize >> activate_zoomworm.bat

REM Create usage examples
echo REM Worm Segmentation Usage Examples > run_examples.bat
echo @echo off >> run_examples.bat
echo call conda activate zoomworm >> run_examples.bat
echo echo 🐛 Worm Segmentation Usage Examples >> run_examples.bat
echo echo ================================== >> run_examples.bat
echo echo. >> run_examples.bat
echo echo 1. Explore your dataset: >> run_examples.bat
echo echo python explore_data.py --visualize >> run_examples.bat
echo echo. >> run_examples.bat
echo echo 2. Train the model ^(quick test^): >> run_examples.bat
echo echo python main.py --epochs 10 --batch-size 4 --no-wandb >> run_examples.bat
echo echo. >> run_examples.bat
echo echo 3. Train the model ^(full training^): >> run_examples.bat
echo echo python main.py --epochs 100 --batch-size 8 >> run_examples.bat
echo echo. >> run_examples.bat
echo echo 4. Run inference on images: >> run_examples.bat
echo echo python main.py --mode inference --checkpoint checkpoints/best-model.ckpt --input-folder /path/to/images --output-folder results >> run_examples.bat
echo echo. >> run_examples.bat
echo echo 5. Process video ^(fastest^): >> run_examples.bat
echo echo python video_inference.py --video /path/to/video.mp4 --checkpoint checkpoints/best-model.ckpt --output video_results --batch-size 32 >> run_examples.bat

echo.
echo [SUCCESS] 🎉 Installation completed successfully!
echo.
echo Next steps:
echo 1. Run: activate_zoomworm.bat to activate the environment
echo 2. Update data paths in config.py
echo 3. Run: python explore_data.py --visualize
echo.
echo For usage examples, run: run_examples.bat

pause 