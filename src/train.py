import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np

from model import create_model, compile_model, get_callbacks, unfreeze_base_model

# Constants
DATA_DIR = Path("data/split")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10     # Reduced epochs because steps_per_epoch is much larger now
FINE_TUNE_EPOCHS = 20
LEARNING_RATE = 1e-3

# MixUp Configuration
MIXUP_ALPHA = 0.2

def mix_up(images, labels, alpha=0.2):
    """
    Applies MixUp augmentation to a batch of images and labels.
    """
    batch_size = tf.shape(images)[0]
    
    # Sample lambda from Beta distribution
    # Note: KerasCV has a MixUp layer, but manual implementation is safer for dependencies
    # Gamma distribution with alpha=beta mimics Beta distribution
    weight = tf.random.gamma([batch_size], alpha, 1.0)
    beta = tf.random.gamma([batch_size], 1.0, alpha)
    gamma = weight / (weight + beta)
    
    # Reshape for broadcasting to image shape
    gamma_images = tf.reshape(gamma, [-1, 1, 1, 1])
    
    # Shuffle the batch
    indices = tf.range(batch_size)
    shuffled_indices = tf.random.shuffle(indices)
    
    images_mix = gamma_images * images + (1 - gamma_images) * tf.gather(images, shuffled_indices)
    
    # Labels mix
    gamma_labels = tf.reshape(gamma, [-1, 1])
    labels_mix = gamma_labels * labels + (1 - gamma_labels) * tf.gather(labels, shuffled_indices)
    
    return images_mix, labels_mix

def create_balanced_dataset(data_dir, batch_size=32):
    """
    Creates a balanced dataset via Oversampling and MixUp.
    """
    data_dir = Path(data_dir)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    datasets = []
    
    print("Creating balanced dataset (Oversampling + MixUp)...")
    for cls in class_names:
        cls_dir = data_dir / cls
        # Create dataset for this class
        ds = keras.utils.image_dataset_from_directory(
            cls_dir,
            labels=None, 
            image_size=IMAGE_SIZE,
            batch_size=None, 
            shuffle=True,
            seed=42,
            verbose=0
        )
        
        # Add label
        label_int = class_names.index(cls)
        # Convert to one-hot
        ds = ds.map(lambda x: (x, tf.one_hot(label_int, len(class_names))))
        
        # Repeat indefinitely to allow sampling
        ds = ds.repeat()
        datasets.append(ds)
        
    # Sample uniformly from all classes
    balanced_ds = tf.data.Dataset.sample_from_datasets(
        datasets,
        weights=[1.0] * len(class_names)
    )
    
    # Batch first
    balanced_ds = balanced_ds.batch(batch_size)
    
    # Apply MixUp
    balanced_ds = balanced_ds.map(
        lambda x, y: mix_up(x, y, alpha=MIXUP_ALPHA),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Prefetch
    balanced_ds = balanced_ds.prefetch(tf.data.AUTOTUNE)
    
    # Calculate steps to ensure we see the majority class fully each epoch
    # Formula: steps * batch_size * (1/num_classes) >= max_class_count
    # Therefore: steps >= (max_class_count * num_classes) / batch_size
    counts = [len(list((data_dir/cls).glob('*.jpg'))) for cls in class_names]
    max_count = max(counts)
    
    # We set steps so that statistically we cover the majority class once per epoch
    steps_per_epoch = int((max_count * len(class_names)) // batch_size)
    
    print(f"Distribution Optimization:")
    print(f"  - Majority Class Count: {max_count}")
    print(f"  - Required Steps/Epoch: {steps_per_epoch}")
    print(f"  - Samples per Epoch: {steps_per_epoch * batch_size}")
    
    return balanced_ds, class_names, steps_per_epoch

def verify_distribution(dataset, class_names, num_batches=5):
    """
    Verifies that the dataset is actually producing a balanced distribution.
    """
    print(f"\nVerifying Batch Distribution (Checking {num_batches} batches)...")
    
    counts = {name: 0 for name in class_names}
    total = 0
    
    # Take a few batches
    for _, labels_batch in dataset.take(num_batches):
        # labels_mix are soft labels due to MixUp, so we take argmax to see dominant class
        # or just sum them up if they are one-hot/mixed
        # Since MixUp is on, labels are float. We can sum probabilities.
        for label in labels_batch:
            # label shape (7,)
            for i, prob in enumerate(label):
                counts[class_names[i]] += float(prob)
            total += 1
            
    print("Effective Class Distribution in Batches:")
    for name, count in counts.items():
        percentage = (count / total) * 100
        print(f"  - {name:<6}: {percentage:.1f}% (Target: {100/len(class_names):.1f}%)")
    print("Distribution verification complete.\n")

def create_val_dataset(data_dir):
    ds = keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    return ds.prefetch(tf.data.AUTOTUNE)

def plot_training_history(history, save_path):
    metrics = ['accuracy', 'loss', 'auc']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, metric in enumerate(metrics):
        if metric in history.history:
            axes[i].plot(history.history[metric], label=f'Train {metric}')
            axes[i].plot(history.history[f'val_{metric}'], label=f'Val {metric}')
            axes[i].set_title(metric.upper())
            axes[i].set_xlabel('Epoch')
            axes[i].legend()
            axes[i].grid(True)
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("=" * 60)
    print("TRAINING SKIN DISEASE CLASSIFIER")
    print("Strategy: EfficientNetB1 + Focal Loss + Oversampling + MixUp + Strong Augmentation")
    print("=" * 60)
    
    # 1. Data Preparation
    train_ds, class_names, steps_per_epoch = create_balanced_dataset(TRAIN_DIR, BATCH_SIZE)
    val_ds = create_val_dataset(VAL_DIR)
    
    # Verify the balanced nature
    verify_distribution(train_ds, class_names)
    
    print(f"\nClasses: {class_names}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Save mapping
    class_mapping = {i: name for i, name in enumerate(class_names)}
    with open(MODEL_DIR / "class_mapping.json", 'w') as f:
        json.dump(class_mapping, f, indent=2)

    # 2. Model Creation
    print("\nInitializing model...")
    model = create_model(num_classes=len(class_names))
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    callbacks = get_callbacks(MODEL_DIR, append=False)
    
    # 3. Phase 1: Train Head
    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING HEAD")
    print("=" * 60 + "\n")
    
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch
    )
    
    # 4. Phase 2: Fine-Tuning
    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNING BACKBONE")
    print("=" * 60 + "\n")
    
    model = unfreeze_base_model(model, num_layers_unfreeze=40)
    
    # Low learning rate for fine-tuning
    model = compile_model(model, learning_rate=1e-5, weight_decay=1e-4) # Reduced weight decay
    
    # Update callbacks for phase 2 (append=True to keep log)
    callbacks = get_callbacks(MODEL_DIR, append=True)
    callbacks[0] = keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_model_finetuned.keras"),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch
    )
    
    plot_training_history(history_2, RESULTS_DIR / "training_history.png")
    print("\nTraining complete. Best model saved to models/best_model_finetuned.keras")

if __name__ == "__main__":
    main()
