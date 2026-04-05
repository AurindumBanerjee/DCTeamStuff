# bash /DATA/anikde/Aurindum/DCTeam/DC_VIT/quickstart-pathb.sh


#!/bin/bash
# QUICK START: PATH B GPU TRAINING
# Run this to get training with GPU support in ~2 minutes

echo "PATH B Quick Start"
echo "=================="
echo ""

# Step 1: Setup environment
echo "[1/3] Installing compatible packages (TensorFlow 2.12.1)..."
bash /DATA/anikde/Aurindum/DCTeam/DC_VIT/setup-pathb.sh bob || {
    echo "Setup failed. Run manually:"
    echo "  conda run -n bob pip install --upgrade --force-reinstall -r requirements-pathb.txt"
    exit 1
}

# Step 2: Quick GPU verification
echo ""
echo "[2/3] Verifying GPU detection..."
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
conda run -n bob python << 'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ SUCCESS: {len(gpus)} GPU(s) detected!")
    for g in gpus:
        print(f"    {g}")
else:
    print("⚠ WARNING: No GPUs detected. Check driver compatibility.")
EOF

# Step 3: Launch training
echo ""
echo "[3/3] Starting training in background..."
nohup bash /DATA/anikde/Aurindum/DCTeam/DC_VIT/train-pathb.sh > /DATA/anikde/Aurindum/DCTeam/DC_VIT/dc-aug-3april-pathb.log 2>&1 &

PID=$!
echo "✓ Training started (PID: $PID)"
echo ""
echo "Monitor progress with:"
echo "  tail -f dc-aug-3april-pathb.log"
echo ""
echo "Kill process with:"
echo "  kill $PID"
