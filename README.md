DC_VIT
Dataset

Current Recommended Execution (this machine)

Use PATH B for the active setup:
- TensorFlow 2.10.1
- Single GPU (GPU 0)
- Training entrypoint: dc-aug-3april-pathb.py

Quick start:
```bash
cd /DATA/anikde/Aurindum/DCTeam/DC_VIT

# one-time only if environment is missing or broken
bash setup-pathb.sh bob

# start training (train-pathb.sh already runs in background with nohup)
bash train-pathb.sh

# find latest run folder and watch logs
LATEST_RUN=$(ls -dt /DATA/anikde/Aurindum/DCTeam/DC_VIT/runs/* | head -n 1)
tail -f "$LATEST_RUN/train.log"
```

What each step achieves:
1. `setup-pathb.sh`: installs and verifies the compatible dependency stack.
2. `train-pathb.sh`: exports CUDA paths, launches training in background, creates a timestamped run folder, and writes logs/checkpoints/plots there.
3. `tail -f`: streams progress from the latest run's log file.

Optional launch settings:
1. `TRAIN_PATHB_ENV=myenv bash train-pathb.sh` uses a different conda env.
2. `TRAIN_PATHB_RUNS_DIR=/path/to/runs bash train-pathb.sh` changes base runs directory.
3. `TRAIN_PATHB_RUN_DIR=/path/to/custom_run bash train-pathb.sh` sets exact run directory.
4. `TRAIN_PATHB_LOG=/path/to/custom.log bash train-pathb.sh` overrides log file path.
5. `bash train-pathb.sh --foreground` runs interactively for debugging.

Detailed guide:
- README-pathb.md

Download the dataset from Google Drive:
https://drive.google.com/drive/folders/1gjdmyTR_9B7U1-W7hWugewnSowjetXYC?usp=drive_link

Use the 12-way script classification dataset

How to Run (Legacy Notebook Flow)
1. Clone the Repository
```
git clone https://github.com/codebythanos/DC_VIT.git
cd DC_VIT
```
2. Install Required Libraries
```
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras jupyter
```
4. Dataset Setup (Important)

Place the dataset in the following structure:

```
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
4. Fix Paths in Notebook

The notebook currently uses Kaggle paths like:
```
train_dir = '/kaggle/input/.../train_1800'
test_dir  = '/kaggle/input/.../test_478'
```
Replace them with:
```
train_dir = 'dataset/12-way script classification dataset/train_1800'
test_dir  = 'dataset/12-way script classification dataset/test_478'
```
5. Run the Notebook

Start Jupyter Notebook:

jupyter notebook

Open:

dc-aug-3april.ipynb

Run all cells using:

Shift + Enter
