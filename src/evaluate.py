"""
Script d'évaluation du modèle de classification sur le jeu de test.
Inclut Test Time Augmentation (TTA).
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score
)
from tqdm import tqdm

# Imports for TTA
from tensorflow.keras import layers

# Configuration
DATA_DIR = Path("data/split")
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TTA_STEPS = 5  # Number of TTA forward passes

# Add Focal Loss to scope for loading
try:
    from tensorflow.keras.losses import CategoricalFocalCrossentropy
except ImportError:
    # Use standard CCE if loading fails locally, but model loading usually needs the class
    # If the custom object was saved with the model, we need to provide it upon loading
    class CategoricalFocalCrossentropy(keras.losses.Loss):
         def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
            self.gamma = gamma
            self.from_logits = from_logits
         def call(self, y_true, y_pred):
            return tf.reduce_sum(y_true, axis=-1) # Dummy implementation for loading

def load_test_data():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Dossier de test non trouvé: {TEST_DIR}")
        
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False 
    )
    return test_ds

def load_model_and_classes():
    model_path = MODEL_DIR / "best_model_finetuned.keras"
    if not model_path.exists():
         model_path = MODEL_DIR / "best_model.keras"
         
    if not model_path.exists():
        raise FileNotFoundError("Aucun modèle trouvé dans models/")
        
    print(f"Chargement du modèle: {model_path}")
    
    # Register custom objects
    custom_objects = {"CategoricalFocalCrossentropy": CategoricalFocalCrossentropy}
    
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Standard loading failed ({e}), trying without custom objects compilation...")
        model = keras.models.load_model(model_path, compile=False)
    
    with open(MODEL_DIR / "class_mapping.json", 'r') as f:
        class_mapping_raw = json.load(f)
        class_mapping = {int(k): v for k, v in class_mapping_raw.items()}
        
    return model, class_mapping

def tta_predict(model, dataset, steps=5):
    """
    Test Time Augmentation:
    Predict 'steps' times on augmented versions of the image and average the results.
    """
    print(f"Performing TTA with {steps} steps...")
    
    # Define TTA augmentation layer (lighter than training)
    tta_layers = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.05),
    ])
    
    # Get all images and labels as arrays first
    all_images = []
    all_labels = []
    
    # Unpack dataset (careful with memory if dataset is >10GB, but HAM10000 test is small)
    for imgs, lbls in dataset:
        all_images.append(imgs.numpy())
        all_labels.append(lbls.numpy())
        
    X_test = np.concatenate(all_images)
    y_true = np.concatenate(all_labels)
    
    # 1. Standard Prediction (No Augmentation)
    print("  Step 0: Standard prediction")
    probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    
    # 2. Augmented Predictions
    for i in range(steps - 1):
        print(f"  Step {i+1}: Augmented prediction")
        # Apply augmentation
        # Note: We need to do this batch-wise or array-wise
        # tf.keras layers can take numpy arrays
        X_aug = tta_layers(X_test, training=True) # training=True activates the layers
        p = model.predict(X_aug, batch_size=BATCH_SIZE, verbose=0)
        probs += p
        
    # Average
    probs /= steps
    
    return probs, y_true

def evaluate_model():
    print("="*60)
    print("ÉVALUATION DU MODÈLE (HOLD-OUT TEST SET)")
    print("="*60)
    
    # 1. Load Data & Model
    test_ds = load_test_data()
    model, class_mapping = load_model_and_classes()
    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    
    # 2. Prediction with TTA
    if TTA_STEPS > 1:
        y_pred_probs, y_true_onehot = tta_predict(model, test_ds, steps=TTA_STEPS)
    else:
        y_pred_probs = model.predict(test_ds, verbose=1)
        y_true_onehot = np.concatenate([y for x, y in test_ds], axis=0)
        
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true_cls = np.argmax(y_true_onehot, axis=1)
    
    # 3. Metrics
    acc = accuracy_score(y_true_cls, y_pred)
    balanced_acc = balanced_accuracy_score(y_true_cls, y_pred)
    
    print("\n" + "="*40)
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("="*40)
    
    print("\nRapport de Classification:")
    print(classification_report(y_true_cls, y_pred, target_names=class_names))
    
    # Save Report
    report = classification_report(y_true_cls, y_pred, target_names=class_names, output_dict=True)
    with open(RESULTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    # 4. Confusion Matrix
    cm = confusion_matrix(y_true_cls, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.title(f'Matrice de Confusion (TTA={TTA_STEPS})')
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    plt.close()
    
    print(f"\nRésultats sauvegardés dans {RESULTS_DIR}")

if __name__ == "__main__":
    evaluate_model()
