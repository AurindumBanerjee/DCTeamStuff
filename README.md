# DC_VIT

Forked repository for the current machine setup.

Current remote:
- `origin` -> `https://github.com/AurindumBanerjee/DCTeamStuff.git`

## What To Use Now

The active execution path on this machine is PATH B:
- TensorFlow 2.10.1
- cuDNN 8.1.x compatible runtime
- ResNet50-based training script: `dc-aug-3april-pathb.py`
- Single-GPU execution on GPU 0
- Timestamped output folders per run

Detailed PATH B notes are also available in [README-pathb.md](README-pathb.md).

## Quick Start

```bash
cd /DATA/anikde/Aurindum/DCTeam/DC_VIT

# one-time only if the environment is missing or broken
bash setup-pathb.sh bob

# start training; the launcher creates a timestamped run folder automatically
bash train-pathb.sh

# watch the newest run log
LATEST_RUN=$(ls -dt /DATA/anikde/Aurindum/DCTeam/DC_VIT/runs/* | head -n 1)
tail -f "$LATEST_RUN/train.log"
```

## What Each Step Does

1. `setup-pathb.sh` prepares or repairs the conda environment and installs the compatible dependency stack.
2. `train-pathb.sh` exports CUDA paths, starts training, and writes logs, checkpoints, and plots into a unique timestamped run folder.
3. `tail -f` shows live progress from the latest run.

## Optional Launch Settings

You can override the launcher behavior when needed:

- `TRAIN_PATHB_ENV=myenv bash train-pathb.sh` uses a different conda environment.
- `TRAIN_PATHB_RUNS_DIR=/path/to/runs bash train-pathb.sh` changes the base directory for run folders.
- `TRAIN_PATHB_RUN_DIR=/path/to/custom_run bash train-pathb.sh` forces a specific run folder.
- `TRAIN_PATHB_LOG=/path/to/custom.log bash train-pathb.sh` overrides the log file path.
- `bash train-pathb.sh --foreground` runs interactively and streams output to the terminal.

## Output Layout

Each launch creates a unique folder like:

`/DATA/anikde/Aurindum/DCTeam/DC_VIT/runs/YYYYMMDD_HHMMSS`

Inside that folder you will find:
- `train.log`
- `p1_*_best.keras`
- `p2_*_cfg*.keras`
- `cm_5k.png`, `cm_10k.png`, `cm_20k.png`, `cm_50k.png`
- `aug_comparison.png`

## Process Checks

```bash
ps -ef | grep -E "dc-aug-3april-pathb.py|train-pathb.sh" | grep -v grep
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

## Do I Still Need `setup-pathb.sh`?

Only when:
- the environment is new
- dependencies are broken or missing
- you want to re-install the PATH B stack cleanly

For everyday reruns, use only:

```bash
cd /DATA/anikde/Aurindum/DCTeam/DC_VIT
bash train-pathb.sh
```

## Legacy Notebook Flow

If you want the older notebook-based workflow, the dataset layout and notebook path are still documented below.

### Dataset
Download the dataset from Google Drive:
https://drive.google.com/drive/folders/1gjdmyTR_9B7U1-W7hWugewnSowjetXYC?usp=drive_link

Use the 12-way script classification dataset.

### Dataset Layout

```text
dataset/
|-- 12-way script classification dataset/
|   |-- train_1800/
|   |   |-- class1/
|   |   |-- class2/
|   |   `-- ...
|   `-- test_478/
|       |-- class1/
|       |-- class2/
|       `-- ...
```

### Notebook Setup

1. Clone the repository:

```bash
git clone https://github.com/AurindumBanerjee/DCTeamStuff.git
cd DC_VIT
```

2. Install the notebook dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras jupyter
```

3. Update notebook paths if needed:

```python
train_dir = 'dataset/12-way script classification dataset/train_1800'
test_dir  = 'dataset/12-way script classification dataset/test_478'
```

4. Launch Jupyter:

```bash
jupyter notebook
```

5. Open `dc-aug-3april.ipynb` and run all cells.

## Notes

- `train-pathb.sh` now creates a new timestamped run directory automatically.
- `nohup` is already handled inside the launcher for background execution.
- The current forked repo is configured for GPU 0 only because of the compatible single-GPU path.
