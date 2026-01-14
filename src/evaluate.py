"""
Script d'évaluation du modèle de classification.
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
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


# Configuration
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def load_model_and_config():
    """
    Charge le modèle entraîné et sa configuration.
    
    Returns:
        model, class_mapping
    """
    # Charger le meilleur modèle
    model_path = MODEL_DIR / "best_model_finetuned.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle non trouvé: {model_path}\n"
            "Veuillez d'abord entraîner le modèle avec: python src/train.py"
        )
    
    print(f"Chargement du modèle: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Charger le mapping des classes
    with open(MODEL_DIR / "class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    
    # Convertir les clés en int
    class_mapping = {int(k): v for k, v in class_mapping.items()}
    
    return model, class_mapping


def create_test_generator():
    """
    Crée le générateur de données pour le test (utilise le split validation).
    
    Returns:
        test_ds
    """
    datagen = keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.2
    )
    
    test_ds = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"Images de test: {test_ds.samples}")
    
    return test_ds


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Affiche et sauvegarde la matrice de confusion.
    
    Args:
        y_true: Labels vrais
        y_pred: Labels prédits
        class_names: Noms des classes
        save_path: Chemin de sauvegarde
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Nombre de prédictions'}
    )
    plt.title('Matrice de Confusion', fontsize=16, fontweight='bold')
    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Matrice de confusion sauvegardée: {save_path}")
    plt.close()


def plot_roc_curves(y_true, y_pred_proba, class_names, save_path):
    """
    Trace les courbes ROC pour chaque classe.
    
    Args:
        y_true: Labels vrais (one-hot encoded)
        y_pred_proba: Probabilités prédites
        class_names: Noms des classes
        save_path: Chemin de sauvegarde
    """
    n_classes = len(class_names)
    
    # Calculer ROC pour chaque classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    
    for i, color in enumerate(colors):
        plt.plot(
            fpr[i], 
            tpr[i], 
            color=color, 
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Hasard (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs', fontsize=12)
    plt.title('Courbes ROC par Classe', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Courbes ROC sauvegardées: {save_path}")
    plt.close()
    
    return roc_auc


def plot_class_performance(report_dict, class_names, save_path):
    """
    Visualise les performances par classe.
    
    Args:
        report_dict: Rapport de classification (dict)
        class_names: Noms des classes
        save_path: Chemin de sauvegarde
    """
    metrics = ['precision', 'recall', 'f1-score']
    scores = {metric: [] for metric in metrics}
    
    for class_name in class_names:
        for metric in metrics:
            scores[metric].append(report_dict[class_name][metric])
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, metric in enumerate(metrics):
        offset = width * (i - 1)
        ax.bar(x + offset, scores[metric], width, label=metric.capitalize())
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performances par Classe', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Performances par classe sauvegardées: {save_path}")
    plt.close()


def visualize_predictions(model, test_ds, class_names, num_images=16):
    """
    Visualise quelques prédictions du modèle.
    
    Args:
        model: Modèle entraîné
        test_ds: Générateur de test
        class_names: Noms des classes
        num_images: Nombre d'images à afficher
    """
    # Récupérer un batch d'images
    images, labels = next(iter(test_ds))
    num_images = min(num_images, len(images))
    
    # Faire les prédictions
    predictions = model.predict(images[:num_images], verbose=0)
    
    # Visualiser
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for i in range(num_images):
        # Dénormaliser l'image pour l'affichage
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min())
        
        true_label = class_names[np.argmax(labels[i])]
        pred_label = class_names[np.argmax(predictions[i])]
        confidence = np.max(predictions[i])
        
        color = 'green' if true_label == pred_label else 'red'
        
        axes[i].imshow(img)
        axes[i].set_title(
            f'Vrai: {true_label}\n'
            f'Prédit: {pred_label}\n'
            f'Confiance: {confidence:.2%}',
            color=color,
            fontsize=10
        )
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "sample_predictions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Exemples de prédictions sauvegardés: {save_path}")
    plt.close()


def evaluate_model():
    """
    Fonction principale d'évaluation.
    """
    print("=" * 60)
    print("ÉVALUATION DU MODÈLE")
    print("=" * 60)
    
    # Charger le modèle et la configuration
    model, class_mapping = load_model_and_config()
    class_names = [class_mapping[i] for i in sorted(class_mapping.keys())]
    
    print(f"\nNombre de classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Créer le générateur de test
    print("\nChargement des données de test...")
    test_ds = create_test_generator()
    
    # Évaluation globale
    print("\n" + "=" * 60)
    print("ÉVALUATION SUR LE JEU DE TEST")
    print("=" * 60)
    
    results = model.evaluate(test_ds, verbose=1)
    metrics_names = model.metrics_names
    
    print("\nRésultats globaux:")
    for name, value in zip(metrics_names, results):
        print(f"  {name}: {value:.4f}")
    
    # Prédictions complètes
    print("\nGénération des prédictions...")
    test_ds.reset()
    y_pred_proba = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_ds.classes
    
    # Matrice de confusion
    print("\nGénération de la matrice de confusion...")
    plot_confusion_matrix(
        y_true, 
        y_pred, 
        class_names,
        RESULTS_DIR / "confusion_matrix.png"
    )
    
    # Rapport de classification
    print("\n" + "=" * 60)
    print("RAPPORT DE CLASSIFICATION")
    print("=" * 60)
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        digits=4
    )
    print("\n" + report)
    
    # Sauvegarder le rapport
    with open(RESULTS_DIR / "classification_report.txt", 'w') as f:
        f.write(report)
    print(f"Rapport sauvegardé: {RESULTS_DIR / 'classification_report.txt'}")
    
    # Rapport détaillé (dict)
    report_dict = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Performances par classe
    print("\nGénération des graphiques de performances...")
    plot_class_performance(
        report_dict,
        class_names,
        RESULTS_DIR / "class_performance.png"
    )
    
    # Courbes ROC
    print("Génération des courbes ROC...")
    y_true_onehot = label_binarize(y_true, classes=range(len(class_names)))
    roc_auc = plot_roc_curves(
        y_true_onehot,
        y_pred_proba,
        class_names,
        RESULTS_DIR / "roc_curves.png"
    )
    
    # Visualiser quelques prédictions
    print("Génération d'exemples de prédictions...")
    visualize_predictions(model, test_ds, class_names)
    
    # Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ DE L'ÉVALUATION")
    print("=" * 60)
    
    print(f"\nAccuracy globale: {report_dict['accuracy']:.4f}")
    print(f"Macro-average F1-score: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted-average F1-score: {report_dict['weighted avg']['f1-score']:.4f}")
    
    print("\nAUC moyen par classe:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {roc_auc[i]:.4f}")
    
    # Sauvegarder les résultats d'évaluation
    eval_results = {
        "accuracy": float(report_dict['accuracy']),
        "macro_f1": float(report_dict['macro avg']['f1-score']),
        "weighted_f1": float(report_dict['weighted avg']['f1-score']),
        "per_class": {
            class_names[i]: {
                "precision": float(report_dict[class_names[i]]['precision']),
                "recall": float(report_dict[class_names[i]]['recall']),
                "f1-score": float(report_dict[class_names[i]]['f1-score']),
                "support": int(report_dict[class_names[i]]['support']),
                "auc": float(roc_auc[i])
            }
            for i in range(len(class_names))
        }
    }
    
    with open(RESULTS_DIR / "evaluation_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nRésultats détaillés sauvegardés: {RESULTS_DIR / 'evaluation_results.json'}")
    print(f"\nTous les graphiques sont dans: {RESULTS_DIR}")


if __name__ == "__main__":
    evaluate_model()
