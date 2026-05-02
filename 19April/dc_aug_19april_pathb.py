#!/usr/bin/env python
# coding: utf-8
#
# ══════════════════════════════════════════════════════════════════════════════
#  PATCH-SIZE EXPERIMENT + ATTENTION HEATMAP ANALYSIS
#  Informed by PARSeq (ECCV 2022) ViT encoder design
#
#  PARSeq: 12-layer ViT, no [CLS] token, no classification head,
#          image 128×32, patch 8×4 → 64 tokens, d_model=384 (PARSeq-S)
#
#  Our 224×224 equivalents:
#    vit_base_patch16_224  → 196 tokens  (standard baseline)
#    vit_base_patch8_224   → 784 tokens  (PARSeq-style dense coverage)
#    vit_base_patch32_224  →  49 tokens  (coarse / fast)
#    custom patch16h×8w    → asymmetric  (PARSeq philosophy on square images)
#
#  After each experiment attention heatmaps are produced so you can see
#  what spatial regions each patch configuration focuses on.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
import warnings, os, sys, copy, cv2

warnings.filterwarnings('ignore')


# ────────────────────────────────────────────────────────────────
# GPU DETECTION
# ────────────────────────────────────────────────────────────────
print("="*70)
print("GPU DETECTION & ENVIRONMENT CHECK")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"Python version:  {sys.version}")

if 'LD_LIBRARY_PATH' in os.environ:
    old_ld = os.environ['LD_LIBRARY_PATH']
    del os.environ['LD_LIBRARY_PATH']
    print(f"Cleared conflicting LD_LIBRARY_PATH: {old_ld}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Physical GPUs detected: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"Using device: {device}")
print("="*70)


# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────
train_dir   = '/DATA/anikde/Aurindum/DCTeam/data/train_1800'
test_dir    = '/DATA/anikde/Aurindum/DCTeam/data/test_478'

IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_CLASSES     = 12
SEED            = 42
LR              = 1e-3
LABEL_SMOOTHING = 0.1
TARGET_PER_CLASS = 5000          # fixed augmentation target for patch experiments
PHASE1_EPOCHS   = 15
PHASE2_EPOCHS   = 10
PATIENCE        = 6

# Number of sample images to use for heatmap generation
HEATMAP_SAMPLES = 8

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

OUTPUT_DIR = Path(os.environ.get('TRAIN_OUTPUT_DIR', '.')).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR = OUTPUT_DIR / 'heatmaps'
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT = str(OUTPUT_DIR / 'best_overall.pt')

def out_path(name):
    return str(OUTPUT_DIR / name)

print(f"Output directory:  {OUTPUT_DIR}")
print(f"Heatmap directory: {HEATMAP_DIR}")


# ────────────────────────────────────────────────────────────────
# PATCH CONFIGURATIONS TO EXPERIMENT WITH
# ────────────────────────────────────────────────────────────────
#
# Inspired directly by PARSeq's encoder: the paper uses a 12-layer ViT
# with rectangular patches matching image aspect ratio. We replicate
# that spirit on square 224×224 images.
#
PATCH_CONFIGS = [
    {
        'name'      : 'patch16',
        'model_name': 'vit_base_patch16_224',
        'patch_h'   : 16,
        'patch_w'   : 16,
        'description': 'Standard 16×16 — 196 tokens — PARSeq-S baseline equivalent',
    },
    {
        'name'      : 'patch8',
        'model_name': 'vit_base_patch8_224',
        'patch_h'   : 8,
        'patch_w'   : 8,
        'description': 'Fine 8×8 — 784 tokens — analogous to PARSeq dense 8×4 coverage',
    },
    {
        'name'      : 'patch32',
        'model_name': 'vit_base_patch32_224',
        'patch_h'   : 32,
        'patch_w'   : 32,
        'description': 'Coarse 32×32 — 49 tokens — fast low-res baseline',
    },
]

# PARSeq uses ASYMMETRIC patches (8 wide × 4 tall on 128×32 images).
# On 224×224 we replicate this philosophy with a custom backbone.
# timm's `vit_base_patch16_224` can be re-instantiated with custom patch size.
PATCH_CONFIGS.append({
    'name'      : 'patch16h_8w',
    'model_name': 'vit_base_patch16_224',   # will be overridden in build_model
    'patch_h'   : 16,
    'patch_w'   : 8,
    'description': 'Asymmetric 16×8 — 392 tokens — direct PARSeq philosophy on square images',
    'asymmetric': True,
})

print(f"\nPatch experiments planned: {len(PATCH_CONFIGS)}")
for cfg in PATCH_CONFIGS:
    tokens = (IMG_SIZE // cfg['patch_h']) * (IMG_SIZE // cfg['patch_w'])
    print(f"  [{cfg['name']:14s}] {cfg['description']} → {tokens} tokens")


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

    idx = np.random.RandomState(SEED).permutation(len(real_paths))
    real_paths  = real_paths[idx]
    real_labels = real_labels[idx]

    if not training:
        test_ds = ImageDataset(real_paths, real_labels, transform=val_transform)
        loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)
        return loader, cls_names, real_paths, real_labels

    split      = int(0.9 * len(real_paths))
    tr_paths   = real_paths[:split];  val_paths  = real_paths[split:]
    tr_labels  = real_labels[:split]; val_labels = real_labels[split:]

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

    class SmartTransformDataset(Dataset):
        def __init__(self, base_ds):
            self.base_ds = base_ds
        def __len__(self):
            return len(self.base_ds)
        def __getitem__(self, idx):
            img = Image.open(self.base_ds.paths[idx]).convert('RGB')
            tfm = train_transform if self.base_ds.aug_flags[idx] == 1 else val_transform
            return tfm(img), self.base_ds.labels[idx]

    base_train_ds = ImageDataset(all_tr_paths, all_tr_labels,
                                  transform=None, aug_flags=aug_flags)
    train_loader = DataLoader(SmartTransformDataset(base_train_ds),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    val_ds     = ImageDataset(val_paths, val_labels, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    return train_loader, val_loader, cls_names, all_tr_labels, val_labels

print("Dataset builder ready")


# ────────────────────────────────────────────────────────────────
# MODEL BUILDER  (PARSeq-informed encoder design)
# ────────────────────────────────────────────────────────────────
#
# PARSeq encoder details (from paper):
#   • 12 transformer layers
#   • No [CLS] token  →  all patch tokens go to decoder
#   • No classification head on encoder
#   • d_model = 384 (PARSeq-S) / 192 (PARSeq-Ti)
#   • Uses mean-pooling of patch tokens for classification (our adaptation)
#
# We add `output_attentions=True` capability via forward hooks so that
# attention weights can be extracted for heatmap generation without
# modifying the timm internals.
#
def build_model(patch_cfg, learning_rate=LR, dropout=0.3,
                trainable_backbone=False, l2_reg=1e-4):

    is_asymmetric = patch_cfg.get('asymmetric', False)

    if is_asymmetric:
        # timm supports custom patch_size as a tuple for asymmetric patches
        backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,           # no pretrained weights for custom patch
            num_classes=0,
            patch_size=(patch_cfg['patch_h'], patch_cfg['patch_w']),
            img_size=IMG_SIZE,
        )
        print(f"  [Asymmetric] Custom patch {patch_cfg['patch_h']}×{patch_cfg['patch_w']} "
              f"— pretrained weights NOT loaded (shape mismatch)")
    else:
        backbone = timm.create_model(
            patch_cfg['model_name'],
            pretrained=True,
            num_classes=0,
        )

    for param in backbone.parameters():
        param.requires_grad = trainable_backbone

    embed_dim = backbone.num_features

    # Classification head — matches PARSeq spirit: LayerNorm → Linear stack
    # PARSeq uses mean of all patch tokens (no CLS); we do the same via
    # timm's global_pool='avg' (set num_classes=0 gives features directly)
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
            # Storage for attention maps extracted via hooks
            self._attn_maps = []
            self._hooks     = []

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

        # ── Attention hook utilities ──────────────────────────────────
        def register_attention_hooks(self):
            """Register forward hooks on every attention block."""
            self._attn_maps = []
            self._hooks     = []

            def make_hook(layer_idx):
                def hook(module, input, output):
                    # timm attention modules return the projected output;
                    # we capture the raw QK softmax by monkey-patching
                    # the internal attn_drop output stored in module.attn_weights
                    if hasattr(module, 'attn_weights'):
                        self._attn_maps.append(
                            module.attn_weights.detach().cpu())
                return hook

            for i, block in enumerate(self.backbone.blocks):
                # Patch timm's Attention.forward to expose weights
                _patch_attention(block.attn)
                h = block.attn.register_forward_hook(make_hook(i))
                self._hooks.append(h)

        def remove_attention_hooks(self):
            for h in self._hooks:
                h.remove()
            self._hooks = []

        def get_attention_maps(self):
            return self._attn_maps

    model = ViTClassifier(backbone, head).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=l2_reg)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    return model, optimizer, criterion


def _patch_attention(attn_module):
    """
    Monkey-patch a timm Attention module so it stores the softmax
    attention weights in `attn_module.attn_weights` after each forward.
    This avoids rewriting timm internals.
    """
    # Avoid wrapping the same module multiple times.
    if getattr(attn_module, '_attn_patched_for_heatmap', False):
        return

    original_forward = attn_module.forward

    def patched_forward(x, *args, **kwargs):
        # Replicate timm attention forward, capturing weights
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(
            B, N, 3, attn_module.num_heads,
            C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale   = attn_module.scale
        attn    = (q @ k.transpose(-2, -1)) * scale
        attn    = attn.softmax(dim=-1)
        attn_module.attn_weights = attn   # ← stored here
        attn    = attn_module.attn_drop(attn)
        x       = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x       = attn_module.proj(x)
        x       = attn_module.proj_drop(x)
        return x

    attn_module.forward = patched_forward
    attn_module._attn_original_forward = original_forward
    attn_module._attn_patched_for_heatmap = True

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
                epochs, tmp_ckpt_name, patience=PATIENCE):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-8, verbose=True)
    best_val_acc  = 0.0
    epochs_no_imp = 0
    tmp_path      = out_path(tmp_ckpt_name)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc   = run_epoch(model, train_loader, criterion, optimizer, True)
        val_loss, val_acc = run_epoch(model, val_loader,   criterion, training=False)
        scheduler.step(val_acc)

        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            torch.save(model.state_dict(), tmp_path)
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(tmp_path, map_location=device))
    return model, best_val_acc


def _cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            os.remove(p)

print("Training helpers ready")


# ────────────────────────────────────────────────────────────────
# ATTENTION HEATMAP GENERATION
# ────────────────────────────────────────────────────────────────
#
# Strategy (follows PARSeq attention analysis conventions):
#   1. Use Attention Rollout (Abnar & Zuidema, 2020):
#      multiply attention maps across layers, accounting for skip connections.
#   2. Average across heads at each layer.
#   3. Reshape the token-level attention back to spatial grid.
#   4. Overlay on original image using a jet colormap.
#
def attention_rollout(attn_maps_list):
    """
    attn_maps_list: list of tensors [B, heads, N, N] for each layer.
    Returns: tensor [B, N, N] — rolled-out attention.
    """
    # Average over heads
    result = attn_maps_list[0].mean(dim=1)          # [B, N, N]
    for attn in attn_maps_list[1:]:
        attn_avg = attn.mean(dim=1)                 # [B, N, N]
        # Add identity (skip connection) and renormalise rows
        attn_aug  = attn_avg + torch.eye(attn_avg.size(-1))
        attn_aug  = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
        result    = torch.bmm(attn_aug, result)
    return result   # [B, N, N]


def generate_heatmaps(model, raw_image_paths, patch_cfg, class_names,
                      n_samples=HEATMAP_SAMPLES, true_labels=None):
    """
    Generate and save attention heatmaps for n_samples images.
    Uses Attention Rollout over all 12 ViT layers.
    """
    patch_h  = patch_cfg['patch_h']
    patch_w  = patch_cfg['patch_w']
    exp_name = patch_cfg['name']

    n_patches_h = IMG_SIZE // patch_h
    n_patches_w = IMG_SIZE // patch_w
    n_tokens    = n_patches_h * n_patches_w

    # Check if backbone has CLS token
    has_cls = hasattr(model.backbone, 'cls_token') and \
              model.backbone.cls_token is not None

    model.eval()
    model.register_attention_hooks()

    sample_indices = np.random.choice(len(raw_image_paths),
                                       min(n_samples, len(raw_image_paths)),
                                       replace=False)

    fig, axes = plt.subplots(n_samples, 3,
                              figsize=(15, 5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row_idx, img_idx in enumerate(sample_indices):
        img_path = raw_image_paths[img_idx]

        # Load original image for display
        orig_img = Image.open(img_path).convert('RGB')
        orig_arr = np.array(orig_img.resize((IMG_SIZE, IMG_SIZE)))

        # Preprocess for model
        tensor = val_transform(orig_img).unsqueeze(0).to(device)

        with torch.no_grad():
            model._attn_maps = []
            _ = model(tensor)
            attn_maps = model.get_attention_maps()   # list of [1, heads, N+1, N+1]

        if not attn_maps:
            print(f"  WARNING: No attention maps captured for {exp_name}. "
                  f"Check that _patch_attention hook is working.")
            continue

        # Stack and apply rollout
        # Each map: [1, heads, N(+1), N(+1)] where +1 is CLS token if present
        rolled = attention_rollout(attn_maps)  # [1, N(+1), N(+1)]

        if has_cls:
            # Attention FROM cls token TO patches
            attn_vec = rolled[0, 0, 1:]   # [N]
        else:
            # Mean attention across all patch tokens
            attn_vec = rolled[0].mean(dim=0)  # [N]
            # Use only patch tokens (exclude CLS if present)
            if attn_vec.shape[0] > n_tokens:
                attn_vec = attn_vec[-n_tokens:]

        attn_vec = attn_vec[:n_tokens]

        # Reshape to spatial grid
        attn_grid = attn_vec.reshape(n_patches_h, n_patches_w).numpy()

        # Upsample to image size
        attn_up = cv2.resize(attn_grid, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_CUBIC)
        attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)

        # ── Last-layer raw attention (for comparison) ─────────────
        last_attn = attn_maps[-1][0].mean(dim=0)   # [N+CLS, N+CLS]
        if has_cls:
            last_raw = last_attn[0, 1:].numpy()
        else:
            last_raw = last_attn.mean(dim=0).numpy()
            if len(last_raw) > n_tokens:
                last_raw = last_raw[-n_tokens:]
        last_raw = last_raw[:n_tokens].reshape(n_patches_h, n_patches_w)
        last_up  = cv2.resize(last_raw, (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_CUBIC)
        last_up  = (last_up - last_up.min()) / (last_up.max() - last_up.min() + 1e-8)

        # ── Overlay ───────────────────────────────────────────────
        heatmap_rollout = (cm.jet(attn_up)[:, :, :3] * 255).astype(np.uint8)
        overlay_rollout = (0.55 * orig_arr + 0.45 * heatmap_rollout).astype(np.uint8)

        heatmap_last = (cm.jet(last_up)[:, :, :3] * 255).astype(np.uint8)
        overlay_last = (0.55 * orig_arr + 0.45 * heatmap_last).astype(np.uint8)

        label_str = class_names[true_labels[img_idx]] if true_labels is not None else ''

        axes[row_idx, 0].imshow(orig_arr)
        axes[row_idx, 0].set_title(f'Original  [{label_str}]', fontsize=9)
        axes[row_idx, 0].axis('off')

        axes[row_idx, 1].imshow(overlay_rollout)
        axes[row_idx, 1].set_title(
            f'Rollout Attention\n{n_patches_h}×{n_patches_w} grid '
            f'({patch_h}×{patch_w} patches)', fontsize=9)
        axes[row_idx, 1].axis('off')

        axes[row_idx, 2].imshow(overlay_last)
        axes[row_idx, 2].set_title('Last-Layer Attention', fontsize=9)
        axes[row_idx, 2].axis('off')

    model.remove_attention_hooks()

    plt.suptitle(
        f'Attention Heatmaps — {exp_name}\n'
        f'Patch: {patch_h}×{patch_w}  |  Tokens: {n_tokens}  |  '
        f'{patch_cfg["description"]}',
        fontsize=12, fontweight='bold')
    plt.tight_layout()

    save_path = str(HEATMAP_DIR / f'heatmap_{exp_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Heatmaps saved → {save_path}")

    # ── Per-layer attention energy plot ───────────────────────────
    # Shows how attention disperses/focuses across the 12 ViT layers.
    model.register_attention_hooks()
    with torch.no_grad():
        model._attn_maps = []
        _ = model(val_transform(
            Image.open(raw_image_paths[sample_indices[0]]).convert('RGB')
        ).unsqueeze(0).to(device))
        layer_attns = model.get_attention_maps()
    model.remove_attention_hooks()

    entropies = []
    for layer_attn in layer_attns:          # [1, heads, N, N]
        avg_h = layer_attn[0].mean(dim=0)   # [N, N]
        # Mean entropy of attention distributions (rows)
        eps     = 1e-9
        entropy = -(avg_h * (avg_h + eps).log()).sum(dim=-1).mean().item()
        entropies.append(entropy)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(range(1, len(entropies)+1), entropies, marker='o', color='steelblue')
    ax2.set_xlabel('Transformer Layer')
    ax2.set_ylabel('Mean Attention Entropy')
    ax2.set_title(
        f'Attention Entropy per Layer — {exp_name}\n'
        f'(higher = more diffuse, lower = more focused)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    entropy_path = str(HEATMAP_DIR / f'entropy_{exp_name}.png')
    plt.savefig(entropy_path, dpi=120, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Entropy plot saved → {entropy_path}")


# ────────────────────────────────────────────────────────────────
# TEST SET
# ────────────────────────────────────────────────────────────────
print("\nLoading test set...")
try:
    test_loader, class_names, test_paths, true_labels = build_dataset(
        test_dir, target_per_class=None, training=False)
    print("Test set ready. Classes:", class_names)
except FileNotFoundError as e:
    print(f"✗ Dataset not found: {e}")
    test_loader, class_names, test_paths, true_labels = None, None, None, None


# ────────────────────────────────────────────────────────────────
# MAIN LOOP — iterate over patch configurations
# ────────────────────────────────────────────────────────────────
all_results          = []
global_best_val_acc  = 0.0
global_best_meta     = {}

print("\nBuilding dataset...")
try:
    train_loader, val_loader, class_names, train_labels, val_labels = build_dataset(
        train_dir, target_per_class=TARGET_PER_CLASS, training=True)
    print(f"Train: {len(train_labels)} | Val: {len(val_labels)}")
    dataset_ok = True
except FileNotFoundError as e:
    print(f"✗ Dataset not found: {e}")
    dataset_ok = False


if dataset_ok:
    for patch_cfg in PATCH_CONFIGS:
        pname = patch_cfg['name']
        print(f"\n{'='*68}")
        print(f"  PATCH EXPERIMENT: {pname}")
        print(f"  {patch_cfg['description']}")
        n_tok = (IMG_SIZE // patch_cfg['patch_h']) * (IMG_SIZE // patch_cfg['patch_w'])
        print(f"  Tokens per image: {n_tok}")
        print(f"{'='*68}")

        # ── Phase 1: Frozen backbone ──────────────────────────────
        print(f"\nPhase 1: Head-only training [{pname}]")
        print("-"*55)
        model, optimizer, criterion = build_model(
            patch_cfg, learning_rate=1e-3, dropout=0.3, trainable_backbone=False)

        p1_tmp = f'_tmp_p1_{pname}.pt'
        model, p1_best = train_model(
            model, optimizer, criterion,
            train_loader, val_loader,
            epochs=PHASE1_EPOCHS,
            tmp_ckpt_name=p1_tmp)
        print(f"\nPhase 1 best val accuracy [{pname}]: {p1_best*100:.2f}%")

        # ── Phase 2: Full fine-tune ───────────────────────────────
        print(f"\nPhase 2: Full fine-tune [{pname}]")
        print("-"*55)

        # Reload from phase 1 checkpoint
        model, _, criterion = build_model(
            patch_cfg, learning_rate=5e-5, dropout=0.35, trainable_backbone=False)
        model.load_state_dict(torch.load(out_path(p1_tmp), map_location=device))

        # Stage A: unfreeze top 25% of backbone
        backbone_layers = list(model.backbone.children())
        unfreeze_from   = int(len(backbone_layers) * 0.75)
        for j, layer in enumerate(backbone_layers):
            for param in layer.parameters():
                param.requires_grad = (j >= unfreeze_from)
        for param in model.head.parameters():
            param.requires_grad = True

        optimizer_a = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-5, weight_decay=1e-4)
        p2a_tmp = f'_tmp_p2a_{pname}.pt'
        model, val_a = train_model(
            model, optimizer_a, criterion,
            train_loader, val_loader,
            epochs=4, tmp_ckpt_name=p2a_tmp, patience=4)

        # Stage B: unfreeze all
        for param in model.parameters():
            param.requires_grad = True
        optimizer_b = optim.Adam(model.parameters(), lr=2.5e-5, weight_decay=1e-4)
        p2b_tmp = f'_tmp_p2b_{pname}.pt'
        model, val_b = train_model(
            model, optimizer_b, criterion,
            train_loader, val_loader,
            epochs=PHASE2_EPOCHS, tmp_ckpt_name=p2b_tmp, patience=4)

        _cleanup(out_path(p2a_tmp))

        best_val_acc = max(p1_best, val_a, val_b)
        print(f"\nBest val accuracy [{pname}]: {best_val_acc*100:.2f}%")

        # ── Promote global best ───────────────────────────────────
        if best_val_acc > global_best_val_acc:
            global_best_val_acc = best_val_acc
            global_best_meta    = {
                'patch_config': pname,
                'description' : patch_cfg['description'],
                'val_acc'     : best_val_acc,
            }
            torch.save(model.state_dict(), BEST_CKPT)
            print(f"  ★ New global best ({best_val_acc*100:.2f}%) → best_overall.pt")

        # ── Test evaluation ───────────────────────────────────────
        test_acc = float('nan')
        if test_loader is not None:
            _, test_acc = run_epoch(model, test_loader, criterion, training=False)
            print(f"Test Accuracy [{pname}]: {test_acc*100:.2f}%")

            model.eval()
            all_preds = []
            with torch.no_grad():
                for imgs, _ in test_loader:
                    preds = model(imgs.to(device)).argmax(1).cpu().numpy()
                    all_preds.extend(preds)
            y_pred  = np.array(all_preds)
            cm_mat  = confusion_matrix(true_labels, y_pred)
            cls_acc = cm_mat.diagonal() / cm_mat.sum(axis=1)

            print(f"\nClass-wise Accuracy [{pname}]:")
            print("-"*40)
            for name, acc in zip(class_names, cls_acc):
                print(f"  {name:25s}: {acc*100:.2f}%")
            print(f"  Mean: {cls_acc.mean()*100:.2f}%")
            print("\nClassification Report:")
            print(classification_report(true_labels, y_pred,
                                        target_names=class_names, digits=4))

            # Confusion matrix
            plt.figure(figsize=(14, 12))
            sns.heatmap(cm_mat, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix — {pname}  '
                      f'(patch {patch_cfg["patch_h"]}×{patch_cfg["patch_w"]})',
                      fontsize=14)
            plt.xlabel('Predicted'); plt.ylabel('True')
            plt.xticks(rotation=45, ha='right'); plt.tight_layout()
            plt.show(); plt.close()

        # ── Attention Heatmaps ────────────────────────────────────
        print(f"\nGenerating attention heatmaps [{pname}]...")
        if test_paths is not None:
            generate_heatmaps(
                model       = model,
                raw_image_paths = test_paths,
                patch_cfg   = patch_cfg,
                class_names = class_names,
                n_samples   = HEATMAP_SAMPLES,
                true_labels = true_labels,
            )
        else:
            print("  Skipping heatmaps — test paths not available.")

        all_results.append({
            'patch_config': pname,
            'patch_h'     : patch_cfg['patch_h'],
            'patch_w'     : patch_cfg['patch_w'],
            'n_tokens'    : n_tok,
            'description' : patch_cfg['description'],
            'val_acc'     : best_val_acc,
            'test_acc'    : test_acc,
        })

        _cleanup(out_path(p1_tmp), out_path(p2b_tmp))


# ────────────────────────────────────────────────────────────────
# RESULTS SUMMARY
# ────────────────────────────────────────────────────────────────
if all_results:
    results_df = pd.DataFrame(all_results).sort_values('val_acc', ascending=False)

    print("\n" + "="*70)
    print("PATCH EXPERIMENT RESULTS SUMMARY")
    print("="*70)
    print(results_df[['patch_config','patch_h','patch_w','n_tokens',
                       'val_acc','test_acc']].to_string(index=False))

    print(f"\nGlobal Best: {global_best_meta}")
    print(f"Checkpoint:  {BEST_CKPT}")

    # ── Comparison chart ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Val accuracy bar
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars = axes[0].bar(results_df['patch_config'],
                       results_df['val_acc'] * 100,
                       color=colors[:len(results_df)])
    axes[0].set_xlabel('Patch Configuration')
    axes[0].set_ylabel('Best Val Accuracy (%)')
    axes[0].set_title('Val Accuracy by Patch Size\n(PARSeq-Informed Configurations)')
    axes[0].set_ylim(50, 100)
    for bar, val in zip(bars, results_df['val_acc']):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f'{val*100:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Tokens vs accuracy scatter
    axes[1].scatter(results_df['n_tokens'], results_df['val_acc'] * 100,
                    s=200, c=colors[:len(results_df)], zorder=5)
    for _, row in results_df.iterrows():
        axes[1].annotate(row['patch_config'],
                         (row['n_tokens'], row['val_acc']*100),
                         textcoords='offset points', xytext=(8, 4), fontsize=9)
    axes[1].set_xlabel('Number of Tokens (patch density)')
    axes[1].set_ylabel('Best Val Accuracy (%)')
    axes[1].set_title('Accuracy vs Token Count\n(patch size trade-off)')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Patch Size Experiment — Script Identification  '
                 '(inspired by PARSeq ViT encoder)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'patch_experiment_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"\nSummary chart saved → {OUTPUT_DIR / 'patch_experiment_summary.png'}")
