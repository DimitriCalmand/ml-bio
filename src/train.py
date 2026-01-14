import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np

from model import create_model, compile_model, get_callbacks


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2


def create_data_generators():
    print("Chargement des données...")
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        validation_split=VALIDATION_SPLIT,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        validation_split=VALIDATION_SPLIT
    )
    
    train_ds = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    val_ds = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    class_names = list(train_ds.class_indices.keys())
    
    print(f"\nNombre de classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(f"Images d'entraînement: {train_ds.samples}")
    print(f"Images de validation: {val_ds.samples}")
    
    class_mapping = {idx: name for name, idx in train_ds.class_indices.items()}
    with open(MODEL_DIR / "class_mapping.json", 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    return train_ds, val_ds, class_names


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique d'entraînement sauvegardé: {save_path}")
    plt.close()


def train_model():
    print("=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION")
    print("=" * 60)
    
    if not DATA_DIR.exists():
        print(f"\nErreur: Le dossier {DATA_DIR} n'existe pas.")
        print("Veuillez d'abord exécuter: python src/download_data.py")
        return
    
    train_ds, val_ds, class_names = create_data_generators()
    
    print("\nCréation du modèle...")
    model = create_model(num_classes=len(class_names))
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    print("\nArchitecture du modèle:")
    model.summary()
    
    callbacks = get_callbacks(MODEL_DIR)
    
    print("\n" + "=" * 60)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 60 + "\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)
    
    final_model_path = MODEL_DIR / "final_model.h5"
    model.save(final_model_path)
    print(f"\nModèle final sauvegardé: {final_model_path}")
    
    plot_training_history(history, RESULTS_DIR / "training_history.png")
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES PERFORMANCES")
    print("=" * 60)
    
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    print(f"\nMeilleure accuracy de validation: {best_val_acc:.4f}")
    print(f"Atteinte à l'epoch: {best_epoch}")
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\nPerformances finales:")
    print(f"  Train accuracy: {final_train_acc:.4f}")
    print(f"  Validation accuracy: {final_val_acc:.4f}")
    
    config = {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "validation_split": VALIDATION_SPLIT,
        "num_classes": len(class_names),
        "class_names": class_names,
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "final_train_accuracy": float(final_train_acc),
        "final_val_accuracy": float(final_val_acc)
    }
    
    with open(MODEL_DIR / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration sauvegardée: {MODEL_DIR / 'training_config.json'}")
    print(f"\nTous les résultats sont dans: {MODEL_DIR} et {RESULTS_DIR}")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU détecté: {len(gpus)} device(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("Pas de GPU détecté, utilisation du CPU")
    
    train_model()
