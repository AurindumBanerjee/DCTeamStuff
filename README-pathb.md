# PATH B: Current Execution Guide (Driver 535, Single GPU)

## Overview
The current working setup uses:
- TensorFlow 2.10.1
- cuDNN 8.1.x compatible runtime
- ResNet50-based training script: `dc-aug-3april-pathb.py`
- Single-GPU execution (GPU 0 only)

This is the stable path for the current machine stack.

---

## Current File Roles
- `dc-aug-3april-pathb.py`: Main training script (single GPU, float32 policy)
- `requirements-pathb.txt`: Pinned dependencies (includes `tensorflow==2.10.1`)
- `setup-pathb.sh`: One-time environment bootstrap/repair script
- `train-pathb.sh`: Runtime launcher with CUDA path exports

---

## Execution Template

### 1. One-time setup (new or broken env only)
```bash
bash /DATA/anikde/Aurindum/DCTeam/DC_VIT/setup-pathb.sh bob
```

### 2. Run training (recommended)
```bash
cd /DATA/anikde/Aurindum/DCTeam/DC_VIT
nohup bash train-pathb.sh > dc-aug-3april-pathb.log 2>&1 &
```

### 3. Monitor progress
```bash
tail -f /DATA/anikde/Aurindum/DCTeam/DC_VIT/dc-aug-3april-pathb.log
```

### 4. Check process status
```bash
ps -ef | grep -E "dc-aug-3april-pathb.py|train-pathb.sh" | grep -v grep
```

### 5. Check GPU usage
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

---

## Where Results Are Saved
All outputs are written in:
`/DATA/anikde/Aurindum/DCTeam/DC_VIT`

Expected artifacts:
- Models: `p1_*_best.keras`, `p2_*_cfg*.keras`
- Confusion matrices: `cm_5k.png`, `cm_10k.png`, `cm_20k.png`, `cm_50k.png`
- Comparison chart: `aug_comparison.png`

---

## Do You Still Need `setup-pathb.sh`?
Use it only when needed:
- Yes, if this is a fresh environment or dependencies changed/broke.
- No, for routine reruns once the environment is already working.

For normal daily runs, just use:
```bash
cd /DATA/anikde/Aurindum/DCTeam/DC_VIT
bash train-pathb.sh
```

---

## Notes
- The script is configured to use GPU 0 only.
- `nohup` + `conda run` may buffer log output; if the log appears quiet while process/GPU usage is active, training can still be running.
