#!/bin/bash
# PATH B INSTALLATION SCRIPT
# Installs compatible dependencies for NVIDIA Driver 535 (CUDA 12.2)

set -e

echo "========================================"
echo "PATH B: GPU Environment Setup"
echo "========================================"
echo ""

ENV_NAME="${1:-bob}"

echo "Target conda environment: $ENV_NAME"
echo ""

# Verify environment exists
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "✗ Environment '$ENV_NAME' not found."
    echo "  Create it with: conda create -n $ENV_NAME python=3.10"
    exit 1
fi

echo "Step 1: Upgrading pip..."
conda run -n "$ENV_NAME" pip install --upgrade pip

echo ""
echo "Step 2: Uninstalling incompatible packages..."
conda run -n "$ENV_NAME" pip uninstall -y tensorflow tensorflow-text keras-hub keras 2>/dev/null || true

echo ""
echo "Step 3: Installing PATH B compatible stack..."
echo "  - TensorFlow 2.12.1 (compatible with CUDA 11.2)"
echo "  - Supporting libraries"

# Try with uv if available (better dependency resolution)
if command -v uv &> /dev/null; then
    echo "  Using uv for installation (faster & better dependency resolution)..."
    conda run -n "$ENV_NAME" uv pip install --force-reinstall -r /DATA/anikde/Aurindum/DCTeam/DC_VIT/requirements-pathb.txt
else
    echo "  Using pip for installation..."
    conda run -n "$ENV_NAME" pip install --force-reinstall -r /DATA/anikde/Aurindum/DCTeam/DC_VIT/requirements-pathb.txt
fi

echo ""
echo "Step 4: Verifying installation..."
conda run -n "$ENV_NAME" python -c "
import tensorflow as tf
import keras
print(f'TensorFlow: {tf.__version__}')
print(f'Keras: {keras.__version__}')
print(f'GPU Available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
print(f'Physical GPUs: {tf.config.list_physical_devices(\"GPU\")}')
" || {
    echo "✗ Verification failed. Check environment setup."
    exit 1
}

echo ""
echo "========================================"
echo "✓ PATH B Installation Complete!"
echo "========================================"
echo ""
echo "To run training with GPU:"
echo "  conda run -n $ENV_NAME env -u LD_LIBRARY_PATH python /DATA/anikde/Aurindum/DCTeam/DC_VIT/dc-aug-3april-pathb.py"
echo ""
echo "Or in background:"
echo "  nohup conda run -n $ENV_NAME env -u LD_LIBRARY_PATH python /DATA/anikde/Aurindum/DCTeam/DC_VIT/dc-aug-3april-pathb.py > dc-aug-3april-pathb.log 2>&1 &"
echo ""
