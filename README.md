# DC_VIT

this is the link to dataset:https://drive.google.com/drive/folders/1gjdmyTR_9B7U1-W7hWugewnSowjetXYC?usp=drive_link
use the 12-way dataset


1. Clone the Repository
  git clone https://github.com/codebythanos/DC_VIT.git
  cd DC_VIT
2. Install Required Libraries
  pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras jupyter
3. Dataset Setup (IMPORTANT)

notebook should have this structure:

dataset/
└── 12-way script classification dataset/
    ├── train_1800/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── test_478/
        ├── class1/
        ├── class2/
        └── ...
Fix Path in Notebook

Currently notebook uses Kaggle paths like:

train_dir = '/kaggle/input/.../train_1800'
test_dir  = '/kaggle/input/.../test_478'

Replace them with:

train_dir = 'dataset/12-way script classification dataset/train_1800'
test_dir  = 'dataset/12-way script classification dataset/test_478'
4. Run the Notebook

Start Jupyter:

jupyter notebook

Open:

dc-aug-3april.ipynb

Run all cells using
