# PATH B: CUDA 12.2 Compatible Version (Current NVIDIA Driver 535)
#
# SETUP INSTRUCTIONS:
# 1. Install compatible packages:
#    conda run -n bob pip install torch torchvision timm scikit-learn pandas matplotlib seaborn
#
# 2. Run with proper environment cleanup:
#    conda run -n bob env -u LD_LIBRARY_PATH python dc-aug-fixed-pathb.py > dc-aug-fixed-pathb.log 2>&1 &
#
# 3. Verify GPU detection before training:
#    conda run -n bob python -c "import torch; print('GPUs:', torch.cuda.device_count())"

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from PIL import Image
import warnings, os, sys, copy

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────
# GPU DETECTION & ENVIRONMENT CHECK
# ────────────────────────────────────────────────────────────────
print("="*70)
print("GPU DETECTION & ENVIRONMENT CHECK")
print("="*70)

print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")

if 'LD_LIBRARY_PATH' in os.environ:
    old_ld = os.environ['LD_LIBRARY_PATH']
    del os.environ['LD_LIBRARY_PATH']
    print(f"Cleared conflicting LD_LIBRARY_PATH: {old_ld}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Physical GPUs detected: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

if torch.cuda.is_available():
    print("✓ GPU 0 selected")
else:
    print("✗ WARNING: No GPUs detected. Falling back to CPU.")

print(f"Using device: {device}")
print("="*70)


# ────────────────────────────────────────────────────────────────
# CONFIG & PATHS
# ────────────────────────────────────────────────────────────────
train_dir   = '/DATA/anikde/Aurindum/DCTeam/data/train_1800'
test_dir    = '/DATA/anikde/Aurindum/DCTeam/data/test_478'

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_CLASSES = 12
SEED        = 42
LR          = 1e-3
LABEL_SMOOTHING = 0.1

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

OUTPUT_DIR = Path(os.environ.get('TRAIN_OUTPUT_DIR', '.')).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

def out_path(name):
    return str(OUTPUT_DIR / name)

print("Config ready")


# ────────────────────────────────────────────────────────────────
# AUGMENTATION
# ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Augmentation pipeline ready")


# ────────────────────────────────────────────────────────────────
# DATASET
# FIX: Split on REAL images FIRST, then augment only train split.
# ────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None, aug_flags=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform
        self.aug_flags = aug_flags if aug_flags is not None else np.zeros(len(paths), dtype=np.int32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_dataset(data_dir, target_per_class=None, training=True):
    data_dir   = Path(data_dir)
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    cls_names  = [d.name for d in class_dirs]
    cls_to_idx = {name: idx for idx, name in enumerate(cls_names)}

    # Collect REAL images only
    real_paths, real_labels = [], []
    for class_dir in class_dirs:
        label = cls_to_idx[class_dir.name]
        paths = [str(p) for p in class_dir.glob('*')
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        for p in paths:
            real_paths.append(p)
            real_labels.append(label)

    real_paths  = np.array(real_paths)
    real_labels = np.array(real_labels)

    # Shuffle real images with fixed seed before any split
    idx = np.random.RandomState(SEED).permutation(len(real_paths))
    real_paths  = real_paths[idx]
    real_labels = real_labels[idx]

    # ── Test mode: no split, no augmentation ─────────────────
    if not training:
        test_ds = ImageDataset(real_paths, real_labels, transform=val_transform)
        loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        return loader, cls_names, real_labels

    # ── Train mode: split on real images first ────────────────
    split      = int(0.9 * len(real_paths))
    tr_paths   = real_paths[:split];  val_paths  = real_paths[split:]
    tr_labels  = real_labels[:split]; val_labels = real_labels[split:]

    # Expand training split with augmented copies
    aug_paths, aug_labels = [], []
    if target_per_class is not None:
        label_to_paths = {}
        for p, l in zip(tr_paths, tr_labels):
            label_to_paths.setdefault(l, []).append(p)
        for label, paths in label_to_paths.items():
            needed = max(0, target_per_class - len(paths))
            for i in range(needed):
                aug_paths.append(paths[i % len(paths)])
                aug_labels.append(label)

    all_tr_paths  = np.concatenate([tr_paths,  aug_paths])
    all_tr_labels = np.concatenate([tr_labels, aug_labels])
    aug_flags     = np.array([0]*len(tr_paths) + [1]*len(aug_paths), dtype=np.int32)

    shuf = np.random.RandomState(SEED).permutation(len(all_tr_paths))
    all_tr_paths  = all_tr_paths[shuf]
    all_tr_labels = all_tr_labels[shuf]
    aug_flags     = aug_flags[shuf]

    for name in cls_names:
        l   = cls_to_idx[name]
        n_r = int(np.sum(tr_labels == l))
        n_a = int(np.sum(all_tr_labels == l)) - n_r
        print(f"  {name}: {n_r} real + {n_a} aug = {n_r+n_a} train | "
              f"{int(np.sum(val_labels==l))} val")

    # aug_flags used to apply correct transform per sample
    train_ds = ImageDataset(all_tr_paths, all_tr_labels, transform=None, aug_flags=aug_flags)

    class SmartTransformDataset(Dataset):
        def __init__(self, base_ds):
            self.base_ds = base_ds
        def __len__(self):
            return len(self.base_ds)
        def __getitem__(self, idx):
            img = Image.open(self.base_ds.paths[idx]).convert('RGB')
            tfm = train_transform if self.base_ds.aug_flags[idx] == 1 else val_transform
            return tfm(img), self.base_ds.labels[idx]

    train_loader = DataLoader(SmartTransformDataset(train_ds),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    val_ds     = ImageDataset(val_paths, val_labels, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, cls_names, all_tr_labels, val_labels

print("Dataset builder ready")


# ────────────────────────────────────────────────────────────────
# MODEL: ViT via timm
# ────────────────────────────────────────────────────────────────
def build_model(learning_rate=LR, dropout=0.3, trainable_backbone=False, l2_reg=1e-4):
    backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

    for param in backbone.parameters():
        param.requires_grad = trainable_backbone

    embed_dim = backbone.num_features

    head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, 512),
        nn.GELU(),
        nn.BatchNorm1d(512),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(dropout * 0.5),
        nn.Linear(256, NUM_CLASSES),
    )

    class ViTClassifier(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head     = head

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    model = ViTClassifier(backbone, head).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=l2_reg)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    return model, optimizer, criterion

print("Model builder ready")


# ────────────────────────────────────────────────────────────────
# TRAIN / EVAL HELPERS
# ────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def train_model(model, optimizer, criterion, train_loader, val_loader,
                epochs, ckpt_name, patience=6):
    """Train loop with early stopping and single best-model checkpoint."""
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                        factor=0.5, patience=3,
                                                        min_lr=1e-8, verbose=True)
    best_val_acc   = 0.0
    best_state     = None
    epochs_no_imp  = 0
    ckpt_path      = out_path(ckpt_name)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc   = run_epoch(model, train_loader, criterion, optimizer, training=True)
        val_loss, val_acc = run_epoch(model, val_loader,   criterion, training=False)
        scheduler.step(val_acc)

        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            # Save only state_dict — much lighter than full model file
            best_state    = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, best_val_acc

print("Training helpers ready")


# ────────────────────────────────────────────────────────────────
# TEST SET (built once, reused for all experiments)
# ────────────────────────────────────────────────────────────────
print("\nLoading test set...")
try:
    test_loader, class_names, true_labels = build_dataset(
        test_dir, target_per_class=None, training=False)
    print("Test set ready. Classes:", class_names)
except FileNotFoundError as e:
    print(f"✗ Dataset not found: {e}")
    test_loader, class_names, true_labels = None, None, None


# ────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ────────────────────────────────────────────────────────────────
aug_targets = [5000, 10000, 20000, 50000]

# FIX: much lower LRs for fine-tuning ViT params.
phase2_configs = [
    {'lr': 1e-5,  'dropout': 0.40},
    {'lr': 5e-6,  'dropout': 0.45},
    {'lr': 2e-5,  'dropout': 0.35},
    {'lr': 1e-5,  'dropout': 0.30},
    {'lr': 5e-6,  'dropout': 0.50},
]

all_experiment_results = []

for target in aug_targets:
    print(f"\n{'='*65}")
    print(f"  EXPERIMENT: {target//1000}K images per class")
    print(f"{'='*65}")

    print(f"\nBuilding dataset ({target} per class)...")
    try:
        train_loader, val_loader, class_names, train_labels, val_labels = build_dataset(
            train_dir, target_per_class=target, training=True)
        print(f"Train: {len(train_labels)} | Val: {len(val_labels)}")
    except FileNotFoundError as e:
        print(f"✗ Dataset not found: {e}. Skipping.")
        continue

    # ── PHASE 1: Frozen backbone, train head only ─────────────
    print(f"\nPhase 1: Training classification head | {target//1000}K")
    print("-"*55)

    model, optimizer, criterion = build_model(
        learning_rate=1e-3, dropout=0.3, trainable_backbone=False)

    model, p1_best = train_model(
        model, optimizer, criterion,
        train_loader, val_loader,
        epochs=15,
        ckpt_name=f'p1_{target//1000}k_best.pt',
        patience=6)
    print(f"\nPhase 1 best val accuracy: {p1_best*100:.2f}%")

    # ── PHASE 2: Gradual unfreeze with low LR ─────────────────
    # Stage A → top 25% of layers only (4 epochs)
    # Stage B → all layers at half the LR (6 epochs)
    print(f"\nPhase 2: Fine-tuning | {target//1000}K")
    print("-"*55)

    best_val_acc = 0.0
    best_cfg_idx = None
    best_config  = None
    best_model   = None
    exp_results  = []

    for i, cfg in enumerate(phase2_configs):
        print(f"\n  Config {i+1}: lr={cfg['lr']} | dropout={cfg['dropout']}")
        print("  " + "-"*45)

        # Reload Phase 1 best weights into a fresh model
        model, _, criterion = build_model(
            learning_rate=cfg['lr'], dropout=cfg['dropout'], trainable_backbone=False)
        model.load_state_dict(torch.load(out_path(f'p1_{target//1000}k_best.pt'), map_location=device))

        # Stage A: unfreeze top 25% of backbone layers
        backbone_layers = list(model.backbone.children())
        unfreeze_from   = int(len(backbone_layers) * 0.75)
        for j, layer in enumerate(backbone_layers):
            for param in layer.parameters():
                param.requires_grad = (j >= unfreeze_from)
        # Head always trainable
        for param in model.head.parameters():
            param.requires_grad = True

        optimizer_a = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['lr'], weight_decay=1e-4)

        model, val_a = train_model(
            model, optimizer_a, criterion,
            train_loader, val_loader,
            epochs=4,
            ckpt_name=f'p2a_{target//1000}k_cfg{i+1}.pt',
            patience=4)

        # Stage B: unfreeze ALL layers at half the LR
        for param in model.parameters():
            param.requires_grad = True

        optimizer_b = optim.Adam(
            model.parameters(), lr=cfg['lr'] / 2, weight_decay=1e-4)

        model, val_b = train_model(
            model, optimizer_b, criterion,
            train_loader, val_loader,
            epochs=6,
            ckpt_name=f'p2b_{target//1000}k_cfg{i+1}.pt',
            patience=4)

        # Clean up stage A checkpoint to save space
        stage_a_path = out_path(f'p2a_{target//1000}k_cfg{i+1}.pt')
        if os.path.exists(stage_a_path):
            os.remove(stage_a_path)

        val_acc = max(val_a, val_b)
        print(f"\n  Config {i+1} best val accuracy: {val_acc*100:.2f}%")

        exp_results.append({
            'target'    : target,
            'config_idx': i + 1,
            'lr'        : cfg['lr'],
            'dropout'   : cfg['dropout'],
            'val_acc'   : val_acc,
            'model_path': out_path(f'p2b_{target//1000}k_cfg{i+1}.pt')
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config  = cfg
            best_cfg_idx = i + 1
            best_model   = copy.deepcopy(model)

    # Clean up non-best p2b checkpoints to save space
    for i in range(len(phase2_configs)):
        if i + 1 != best_cfg_idx:
            p = out_path(f'p2b_{target//1000}k_cfg{i+1}.pt')
            if os.path.exists(p):
                os.remove(p)

    # ── Test evaluation ───────────────────────────────────────
    if test_loader is not None:
        print(f"\n{'='*55}")
        print(f"Best config for {target//1000}K → Config {best_cfg_idx}: {best_config}")
        print(f"Best val accuracy: {best_val_acc*100:.2f}%")

        _, test_acc = run_epoch(best_model, test_loader, criterion, training=False)
        print(f"Test Accuracy ({target//1000}K): {test_acc*100:.2f}%")

        best_model.eval()
        all_preds = []
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                preds = best_model(imgs).argmax(1).cpu().numpy()
                all_preds.extend(preds)

        y_pred  = np.array(all_preds)
        cm      = confusion_matrix(true_labels, y_pred)
        cls_acc = cm.diagonal() / cm.sum(axis=1)

        print(f"\nClass-wise Accuracy ({target//1000}K):")
        print("-"*35)
        for name, acc in zip(class_names, cls_acc):
            print(f"  {name:25s}: {acc*100:.2f}%")
        print(f"  Mean: {cls_acc.mean()*100:.2f}%")

        print("\nClassification Report:")
        print(classification_report(true_labels, y_pred,
                                    target_names=class_names, digits=4))

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix — {target//1000}K (Best Config {best_cfg_idx})', fontsize=14)
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(out_path(f'cm_{target//1000}k.png'), dpi=150)
        plt.close()

        for r in exp_results:
            r['test_acc'] = test_acc if r['config_idx'] == best_cfg_idx else np.nan
    else:
        print("Skipping test evaluation (dataset not available)")

    all_experiment_results.extend(exp_results)

    # Clean up Phase 1 checkpoint after all Phase 2 configs done
    p1_path = out_path(f'p1_{target//1000}k_best.pt')
    if os.path.exists(p1_path):
        os.remove(p1_path)

print("\n\nAll experiments complete!")


# ────────────────────────────────────────────────────────────────
# RESULTS SUMMARY
# ────────────────────────────────────────────────────────────────
if all_experiment_results:
    results_df = pd.DataFrame(all_experiment_results)

    print("\n" + "="*70)
    print("FULL RESULTS SUMMARY")
    print("="*70)
    print(results_df[['target','config_idx','lr','dropout','val_acc','test_acc']].to_string(index=False))

    print("\n" + "="*70)
    print("BEST CONFIG PER AUGMENTATION SIZE")
    print("="*70)
    best_per_target = results_df.loc[results_df.groupby('target')['val_acc'].idxmax()]
    for _, row in best_per_target.iterrows():
        print(f"  {int(row['target'])//1000}K | Config {int(row['config_idx'])} "
              f"| lr={row['lr']} | dropout={row['dropout']} "
              f"| val={row['val_acc']*100:.2f}% | test={row['test_acc']*100:.2f}%")

    best_row = results_df.loc[results_df['val_acc'].idxmax()]
    print(f"\nOverall Best:")
    print(f"  {int(best_row['target'])//1000}K | Config {int(best_row['config_idx'])} "
          f"| lr={best_row['lr']} | dropout={best_row['dropout']} "
          f"| val_acc={best_row['val_acc']*100:.2f}%")

    pivot = results_df.groupby('target')['val_acc'].max().reset_index()
    plt.figure(figsize=(8, 5))
    plt.bar([f"{t//1000}K" for t in pivot['target']], pivot['val_acc']*100, color='steelblue')
    plt.xlabel('Images per class'); plt.ylabel('Best Val Accuracy (%)')
    plt.title('Best Val Accuracy vs Augmentation Size')
    plt.ylim(50, 100); plt.tight_layout()
    plt.savefig(out_path('aug_comparison.png'), dpi=150)
    plt.close()