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
- `train-pathb.sh`: Runtime launcher with CUDA path exports and background logging

---

## Execution Template

### 1. One-time setup (new or broken env only)
```bash
bash /DATA/anikde/Aurindum/DCTeam/DC_VIT/setup-pathb.sh bob
```

### 2. Run training (recommended)
```bash
cd /DATA/anikde/Aurindum/DCTeam/DC_VIT
bash train-pathb.sh
```

### 3. Monitor progress
```bash
tail -f /DATA/anikde/Aurindum/DCTeam/DC_VIT/runs/<timestamp>/train.log
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

## What Each Section Achieves

### Setup section
Prepares the environment with compatible versions and verifies TensorFlow can see GPUs.

### Run section
Starts the training job via a stable launcher that sets CUDA-related paths, creates a unique timestamped run folder, and writes logs there.

### Monitor section
Lets you observe progress, detect stalls, and confirm active epochs/configurations.

### Process/GPU check section
Confirms that the job is alive and that GPU memory/compute are actually being used.

### Results section
Defines where models and plots are saved so you can collect final artifacts quickly.

---

## Where Results Are Saved
Each launch creates a unique run folder:
`/DATA/anikde/Aurindum/DCTeam/DC_VIT/runs/YYYYMMDD_HHMMSS`

Expected artifacts inside that run folder:
- Log: `train.log`
- Models: `p1_*_best.keras`, `p2_*_cfg*.keras`
- Confusion matrices: `cm_5k.png`, `cm_10k.png`, `cm_20k.png`, `cm_50k.png`
- Comparison chart: `aug_comparison.png`

---

## Launcher Optionalities

### Foreground debug mode
```bash
cd /DATA/anikde/Aurindum/DCTeam/DC_VIT
bash train-pathb.sh --foreground
```

### Use a different conda environment
```bash
TRAIN_PATHB_ENV=myenv bash /DATA/anikde/Aurindum/DCTeam/DC_VIT/train-pathb.sh
```

### Change base runs directory
```bash
TRAIN_PATHB_RUNS_DIR=/path/to/runs bash /DATA/anikde/Aurindum/DCTeam/DC_VIT/train-pathb.sh
```

### Force an exact run directory
```bash
TRAIN_PATHB_RUN_DIR=/path/to/runs/manual_run_01 bash /DATA/anikde/Aurindum/DCTeam/DC_VIT/train-pathb.sh
```

### Override log file path
```bash
TRAIN_PATHB_LOG=/path/to/custom.log bash /DATA/anikde/Aurindum/DCTeam/DC_VIT/train-pathb.sh
```

---

Use setup-pathb.sh only when needed:
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
