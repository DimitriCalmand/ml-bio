import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import json

from model import create_model, compile_model, get_callbacks

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 2
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

def train_quick_test():
    print("=" * 60)
    print("TEST RAPIDE D'ENTRAÎNEMENT (2 EPOCHS)")
    print("=" * 60)
    
    if not DATA_DIR.exists():
        print(f"\nErreur: Le dossier {DATA_DIR} n'existe pas.")
        return
    
    train_ds, val_ds, class_names = create_data_generators()
    
    print("\nCréation du modèle...")
    model = create_model(num_classes=len(class_names))
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    print(f"\nParamètres entraînables: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    print("\n" + "=" * 60)
    print("DÉBUT DE L'ENTRAÎNEMENT (TEST RAPIDE)")
    print("=" * 60 + "\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("TEST TERMINÉ")
    print("=" * 60)
    
    print(f"\nAccuracy finale:")
    print(f"  Train: {history.history['accuracy'][-1]:.4f}")
    print(f"  Validation: {history.history['val_accuracy'][-1]:.4f}")
    
    print("\n✓ Le système fonctionne correctement!")
    print("Pour un entraînement complet, utilisez: python src/train.py")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU détecté: {len(gpus)} device(s)")
    else:
        print("CPU utilisé")
    
    train_quick_test()
