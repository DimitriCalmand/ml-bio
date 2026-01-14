import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np
from sklearn.utils import class_weight

from model import create_model, compile_model, get_callbacks, unfreeze_base_model


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30  # Epochs initiales (tête seulement)
FINE_TUNE_EPOCHS = 30  # Epochs de fine-tuning
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
    print("ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION (AMÉLIORÉ)")
    print("=" * 60)
    
    if not DATA_DIR.exists():
        print(f"\nErreur: Le dossier {DATA_DIR} n'existe pas.")
        print("Veuillez d'abord exécuter: python src/download_data.py")
        return
    
    # 1. Préparation des données
    train_ds, val_ds, class_names = create_data_generators()
    
    # Calcul des poids de classes pour gérer le déséquilibre
    print("\nCalcul des poids de classes...")
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_ds.classes),
        y=train_ds.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Poids: {class_weights_dict}")
    
    # 2. Création et compilation du modèle
    print("\nCréation du modèle...")
    model = create_model(num_classes=len(class_names))
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    print("\nArchitecture du modèle:")
    model.summary()
    
    callbacks = get_callbacks(MODEL_DIR)
    
    # 3. Phase 1 : Entraînement Initial (Tête seulement)
    print("\n" + "=" * 60)
    print("PHASE 1 : ENTRAÎNEMENT INITIAL (Transfer Learning)")
    print("=" * 60 + "\n")
    
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # 4. Phase 2 : Fine-Tuning
    print("\n" + "=" * 60)
    print("PHASE 2 : FINE-TUNING (Décongélation partielle)")
    print("=" * 60 + "\n")
    
    model = unfreeze_base_model(model, num_layers_unfreeze=30)
    
    # Recompiler avec un learning rate très faible
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Adapter les callbacks pour la phase 2
    # On change le nom du fichier de checkpoint pour ne pas écraser le meilleur de la phase 1
    callbacks[0] = keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_model_finetuned.keras"),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    total_epochs = EPOCHS + FINE_TUNE_EPOCHS
    
    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=history_1.epoch[-1] + 1,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)
    
    # Sauvegarde du modèle final
    final_model_path = MODEL_DIR / "final_model_refined.keras"
    model.save(final_model_path)
    print(f"\nModèle final sauvegardé: {final_model_path}")
    
    # Fusionner l'historique pour le traçage
    history_map = {}
    
    # Fonction pour normaliser les noms de métriques (ex: precision_1 -> precision)
    def normalize_metric_name(name):
        if 'accuracy' in name: return 'accuracy' if 'val' not in name else 'val_accuracy'
        if 'loss' in name: return 'loss' if 'val' not in name else 'val_loss'
        if 'precision' in name: return 'precision' if 'val' not in name else 'val_precision'
        if 'recall' in name: return 'recall' if 'val' not in name else 'val_recall'
        return name

    # Initialiser avec l'historique 1
    for k, v in history_1.history.items():
        std_name = normalize_metric_name(k)
        history_map[std_name] = v

    # Ajouter l'historique 2 en mappant les noms
    for k, v in history_2.history.items():
        std_name = normalize_metric_name(k)
        if std_name in history_map:
            history_map[std_name] = history_map[std_name] + v
        else:
            history_map[std_name] = v
        
    # Créer un objet dummy pour passer à la fonction plot
    class HistoryDummy:
        def __init__(self, history):
            self.history = history
            
    full_history = HistoryDummy(history_map)
    
    plot_training_history(full_history, RESULTS_DIR / "training_history_full.png")
    
    # Mise à jour du fichier de config
    best_val_acc = max(full_history.history['val_accuracy'])
    
    config = {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "initial_epochs": EPOCHS,
        "finetune_epochs": FINE_TUNE_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "finetune_lr": 1e-5,
        "num_classes": len(class_names),
        "class_names": class_names,
        "best_val_accuracy": float(best_val_acc),
        "class_weights": {k: float(v) for k, v in class_weights_dict.items()}
    }
    
    with open(MODEL_DIR / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration sauvegardée: {MODEL_DIR / 'training_config.json'}")
    print(f"Meilleur modèle fine-tuné disponible à: {MODEL_DIR / 'best_model_finetuned.keras'}")
    print(f"Modèle de la phase 1 disponible à: {MODEL_DIR / 'best_model.keras'}")
    
    # Mettre à jour best_model.keras avec le meilleur des deux phases
    best_phase1 = max(history_1.history['val_accuracy'])
    best_phase2 = max(history_2.history['val_accuracy'])
    
    if best_phase2 > best_phase1:
        print("\nLe modèle fine-tuné est meilleur. Mise à jour de best_model.keras...")
        import shutil
        shutil.copy(MODEL_DIR / "best_model_finetuned.keras", MODEL_DIR / "best_model.keras")


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
