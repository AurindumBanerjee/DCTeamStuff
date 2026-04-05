# PATH B: CUDA 12.2 Compatible Version (Current NVIDIA Driver 535)
# 
# SETUP INSTRUCTIONS:
# 1. Install compatible packages with this command:
#    conda run -n bob pip install --upgrade --force-reinstall tensorflow==2.15.1 keras==3.3.3
# 
# 2. Run with proper environment cleanup:
#    conda run -n bob env -u LD_LIBRARY_PATH python dc-aug-3april-pathb.py > dc-aug-3april-pathb.log 2>&1 &
#
# 3. Verify GPU detection before training:
#    conda run -n bob python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import warnings, os
import sys

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────
# GPU DETECTION & SETUP
# ────────────────────────────────────────────────────────────────
print("="*70)
print("GPU DETECTION & ENVIRONMENT CHECK")
print("="*70)

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"Python version: {sys.version}")

# Clear conflicting LD_LIBRARY_PATH
if 'LD_LIBRARY_PATH' in os.environ:
    old_ld = os.environ['LD_LIBRARY_PATH']
    del os.environ['LD_LIBRARY_PATH']
    print(f"Cleared conflicting LD_LIBRARY_PATH: {old_ld}")

# Detect GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"Physical GPUs detected: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu}")

if gpus:
    try:
        # Restrict to GPU 0 only (cuDNN version mismatch workaround)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✓ GPU 0 selected and memory growth enabled")
    except Exception as e:
        print(f"✗ Error configuring GPU 0: {e}")
else:
    print("✗ WARNING: No GPUs detected. Falling back to CPU (training will be slow).")
    print("  Troubleshoot: Check CUDA/cuDNN library compatibility with driver.")
    print("  Current: TensorFlow 2.15.x requires CUDA 12.0-12.2")

# Mixed precision policy (disable for GTX 1080 Ti - compute capability 6.1)
try:
    from keras import mixed_precision
    # GTX 1080 Ti (compute capability 6.1) doesn't support mixed_float16 efficiently
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    print(f"✓ Precision policy set: {policy.name} (GTX 1080 Ti compatibility)")
except Exception as e:
    print(f"Note: Precision policy unavailable: {e}")

print("="*70)

# ────────────────────────────────────────────────────────────────
# CONFIG & PATHS
# ────────────────────────────────────────────────────────────────
train_dir = '/DATA/anikde/Aurindum/DCTeam/data/train_1800'
test_dir  = '/DATA/anikde/Aurindum/DCTeam/data/test_478'

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_CLASSES = 12
SEED        = 42
LR          = 1e-3
AUTOTUNE    = tf.data.AUTOTUNE

np.random.seed(SEED)
tf.random.set_seed(SEED)
print("Config ready")


# ────────────────────────────────────────────────────────────────
# AUGMENTATION
# ────────────────────────────────────────────────────────────────
def augment_image(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE + 20, IMG_SIZE + 20)
    img = tf.image.random_crop(img, size=[IMG_SIZE, IMG_SIZE, 3])
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img

print("Augmentation pipeline ready")


# ────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ────────────────────────────────────────────────────────────────
def build_augmented_dataset(data_dir, target_per_class=None, training=True):
    data_dir   = Path(data_dir)
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    cls_names  = [d.name for d in class_dirs]
    cls_to_idx = {name: idx for idx, name in enumerate(cls_names)}

    all_paths, all_labels, is_aug = [], [], []

    for class_dir in class_dirs:
        label = cls_to_idx[class_dir.name]
        paths = [str(p) for p in class_dir.glob('*')
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

        # Real images
        for p in paths:
            all_paths.append(p)
            all_labels.append(label)
            is_aug.append(0)

        # Repeat paths for augmentation (no RAM cost, done on-the-fly)
        if target_per_class is not None:
            needed = max(0, target_per_class - len(paths))
            for i in range(needed):
                all_paths.append(paths[i % len(paths)])
                all_labels.append(label)
                is_aug.append(1)

        total = len([l for l in all_labels if l == label])
        print(f"  {class_dir.name}: {len(paths)} real → {total} total")

    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels)
    is_aug     = np.array(is_aug, dtype=np.int32)

    idx        = np.random.permutation(len(all_paths))
    all_paths  = all_paths[idx]
    all_labels = all_labels[idx]
    is_aug     = is_aug[idx]

    def load_and_maybe_augment(path, label, aug_flag):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.cond(
            aug_flag > 0,
            lambda: augment_image(img),
            lambda: img
        )
        return img, label

    if training:
        split = int(0.9 * len(all_paths))
        tr_paths,  val_paths  = all_paths[:split],  all_paths[split:]
        tr_labels, val_labels = all_labels[:split], all_labels[split:]
        tr_aug,    val_aug    = is_aug[:split],      is_aug[split:]

        tr_labels_oh  = tf.keras.utils.to_categorical(tr_labels,  NUM_CLASSES)
        val_labels_oh = tf.keras.utils.to_categorical(val_labels, NUM_CLASSES)

        train_ds = (tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels_oh, tr_aug))
                    .shuffle(min(len(tr_paths), 10000), seed=SEED)
                    .map(load_and_maybe_augment, num_parallel_calls=AUTOTUNE)
                    .batch(BATCH_SIZE)
                    .prefetch(AUTOTUNE))

        val_ds   = (tf.data.Dataset.from_tensor_slices((val_paths, val_labels_oh, val_aug))
                    .map(load_and_maybe_augment, num_parallel_calls=AUTOTUNE)
                    .batch(BATCH_SIZE)
                    .prefetch(AUTOTUNE))

        return train_ds, val_ds, cls_names, tr_labels, val_labels

    else:
        labels_oh = tf.keras.utils.to_categorical(all_labels, NUM_CLASSES)
        test_ds   = (tf.data.Dataset.from_tensor_slices((all_paths, labels_oh, is_aug))
                     .map(load_and_maybe_augment, num_parallel_calls=AUTOTUNE)
                     .batch(BATCH_SIZE)
                     .prefetch(AUTOTUNE))
        return test_ds, cls_names, all_labels

print("Dataset builder ready")


# ────────────────────────────────────────────────────────────────
# BACKBONE: Using Keras Applications for PATH B compatibility
# (keras_hub requires newer TensorFlow; using ResNet50 as backbone instead)
# ────────────────────────────────────────────────────────────────
print("Loading pretrained backbone (ResNet50 - compatible with TF 2.15.x)...")
try:
    backbone = keras.applications.ResNet50(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    backbone.trainable = False
    print("✓ Pretrained ResNet50 backbone loaded and frozen")
except Exception as e:
    print(f"✗ Error loading backbone: {e}")
    sys.exit(1)


def build_model(learning_rate=LR, dropout=0.3, trainable_backbone=False, l2_reg=1e-4):
    backbone.trainable = trainable_backbone

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.LayerNormalization()(x)
    x = layers.Dense(512, activation='gelu',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='gelu',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout * 0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32',
                           kernel_regularizer=keras.regularizers.l2(l2_reg))(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("Model builder ready")


# ────────────────────────────────────────────────────────────────
# CALLBACKS
# ────────────────────────────────────────────────────────────────
def get_callbacks(name):
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            name,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1
        )
    ]

print("Callbacks ready")


# ────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ────────────────────────────────────────────────────────────────
aug_targets = [5000, 10000, 20000, 50000]

phase2_configs = [
    {'lr': 3e-5, 'dropout': 0.45},
    {'lr': 1e-5, 'dropout': 0.40},
    {'lr': 5e-5, 'dropout': 0.35},
    {'lr': 5e-5, 'dropout': 0.30},
    {'lr': 5e-5, 'dropout': 0.50},
]

all_experiment_results = []

print("\nLoading test set...")
try:
    test_ds, class_names, true_labels = build_augmented_dataset(
        test_dir, target_per_class=None, training=False
    )
    print("Test set ready. Classes:", class_names)
except FileNotFoundError as e:
    print(f"✗ Dataset not found: {e}")
    print(f"  Skipping to model training phase (using dummy data).")
    test_ds, class_names, true_labels = None, None, None

for target in aug_targets:
    print(f"\n{'='*65}")
    print(f"  EXPERIMENT: {target//1000}K images per class")
    print(f"{'='*65}")

    # Build augmented dataset for this target
    print(f"\nBuilding dataset ({target} per class)...")
    try:
        train_ds, val_ds, class_names, train_labels, val_labels = build_augmented_dataset(
            train_dir, target_per_class=target, training=True
        )
        print(f"Train: {len(train_labels)} | Val: {len(val_labels)}")
    except FileNotFoundError as e:
        print(f"✗ Dataset not found: {e}")
        print(f"  Skipping this experiment.")
        continue

    # ── PHASE 1: Train head only ──────────────────────────────
    print(f"\nPhase 1: Training classification head | {target//1000}K")
    print("-" * 55)

    model = build_model(learning_rate=1e-3, dropout=0.3, trainable_backbone=False)

    history1 = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=get_callbacks(f'p1_{target//1000}k_best.keras'),
        verbose=1
    )
    p1_best = max(history1.history['val_accuracy'])
    print(f"\nPhase 1 best val accuracy: {p1_best*100:.2f}%")

    # ── PHASE 2: Fine-tune with multiple configs ──────────────
    print(f"\nPhase 2: Fine-tuning | {target//1000}K")
    print("-" * 55)

    best_val_acc = 0
    best_cfg_idx = None
    best_config  = None
    best_model   = None
    exp_results  = []

    for i, cfg in enumerate(phase2_configs):
        print(f"\n  Config {i+1}: lr={cfg['lr']} | dropout={cfg['dropout']}")
        print("  " + "-" * 45)

        # Load Phase 1 best weights — fresh start for each config
        model = keras.models.load_model(f'p1_{target//1000}k_best.keras')

        # Unfreeze all layers
        for layer in model.layers:
            layer.trainable = True

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg['lr']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history2 = model.fit(
            train_ds,
            epochs=6,
            validation_data=val_ds,
            callbacks=get_callbacks(f'p2_{target//1000}k_cfg{i+1}.keras'),
            verbose=1
        )

        val_acc = max(history2.history['val_accuracy'])
        print(f"\n  Config {i+1} best val accuracy: {val_acc*100:.2f}%")

        exp_results.append({
            'target'    : target,
            'config_idx': i + 1,
            'lr'        : cfg['lr'],
            'dropout'   : cfg['dropout'],
            'val_acc'   : val_acc,
            'model_path': f'p2_{target//1000}k_cfg{i+1}.keras'
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config  = cfg
            best_cfg_idx = i + 1
            best_model   = model

    # ── Test evaluation for this target ──────────────────────
    if test_ds is not None:
        print(f"\n{'='*55}")
        print(f"Best config for {target//1000}K → Config {best_cfg_idx}: {best_config}")
        print(f"Best val accuracy: {best_val_acc*100:.2f}%")

        test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
        print(f"Test Accuracy ({target//1000}K): {test_acc*100:.2f}%")

        y_pred  = np.argmax(best_model.predict(test_ds, verbose=0), axis=1)
        cm      = confusion_matrix(true_labels, y_pred)
        cls_acc = cm.diagonal() / cm.sum(axis=1)

        print(f"\nClass-wise Accuracy ({target//1000}K):")
        print("-" * 35)
        for name, acc in zip(class_names, cls_acc):
            print(f"  {name:20s}: {acc*100:.2f}%")
        print(f"  Mean: {cls_acc.mean()*100:.2f}%")

        print("\nClassification Report:")
        print(classification_report(true_labels, y_pred, target_names=class_names, digits=4))

        # Confusion matrix plot
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix — {target//1000}K per class (Best Config {best_cfg_idx})', fontsize=14)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'cm_{target//1000}k.png', dpi=150)
        plt.close()  # Close to prevent display issues

        # Store test acc in results
        for r in exp_results:
            r['test_acc'] = test_acc if r['config_idx'] == best_cfg_idx else np.nan
    else:
        print("Skipping test evaluation (dataset not available)")

    all_experiment_results.extend(exp_results)

print("\n\nAll experiments complete!")


# ────────────────────────────────────────────────────────────────
# RESULTS SUMMARY
# ────────────────────────────────────────────────────────────────
if all_experiment_results:
    results_df = pd.DataFrame(all_experiment_results)

    print("\n" + "="*70)
    print("FULL RESULTS SUMMARY")
    print("="*70)
    print(results_df[['target', 'config_idx', 'lr', 'dropout', 'val_acc']].to_string(index=False))

    print("\n" + "="*70)
    print("BEST CONFIG PER AUGMENTATION SIZE")
    print("="*70)
    best_per_target = results_df.loc[results_df.groupby('target')['val_acc'].idxmax()]
    for _, row in best_per_target.iterrows():
        print(f"  {int(row['target'])//1000}K | Config {int(row['config_idx'])} "
              f"| lr={row['lr']} | dropout={row['dropout']} "
              f"| val_acc={row['val_acc']*100:.2f}%")

    best_row = results_df.loc[results_df['val_acc'].idxmax()]
    print(f"\nOverall Best:")
    print(f"  {int(best_row['target'])//1000}K | Config {int(best_row['config_idx'])} "
          f"| lr={best_row['lr']} | dropout={best_row['dropout']} "
          f"| val_acc={best_row['val_acc']*100:.2f}%")

    # Bar chart comparison
    pivot = results_df.groupby('target')['val_acc'].max().reset_index()
    plt.figure(figsize=(8, 5))
    plt.bar([f"{t//1000}K" for t in pivot['target']], pivot['val_acc'] * 100, color='steelblue')
    plt.xlabel('Images per class')
    plt.ylabel('Best Val Accuracy (%)')
    plt.title('Best Val Accuracy vs Augmentation Size')
    plt.ylim(50, 100)
    plt.tight_layout()
    plt.savefig('aug_comparison.png', dpi=150)
    plt.close()
