# Classification des Maladies de Peau - HAM10000

Application de classification automatique des maladies de peau utilisant le dataset HAM10000. Ce projet fournit une interface intelligente pour le diagnostic assisté par ordinateur, optimisée pour la sécurité des patients et expliquée par des techniques d'analyse visuelle (XAI).

## Description

Le projet exploite une architecture **EfficientNetB1** (Transfer Learning) pour classifier 7 types de lésions cutanées. Les modèles sont entraînés avec des techniques avancées pour gérer le fort déséquilibre des données (Focal Loss, MixUp, Oversampling).

| Code | Maladie | Type | Risque |
|------|---------|------|--------|
| **mel** | Melanoma | 🔴 Cancéreux | Critique |
| **bcc** | Basal Cell Carcinoma | 🔴 Cancéreux | Critique |
| **akiec** | Actinic Keratosis | 🟠 Pré-cancéreux | Critique |
| **bkl** | Benign Keratosis | 🟢 Bénin | Bas |
| **nv** | Melanocytic Nevus | 🟢 Bénin | Bas |
| **df** | Dermatofibroma | 🟢 Bénin | Bas |
| **vasc** | Vascular Lesion | 🟢 Bénin | Bas |

## Fonctionnalités Clés

- 🧠 **Modèle Robuste** : EfficientNetB1 fine-tuné avec augmentations fortes (TTA, MixUp).
- 🖥️ **Application Streamlit** : Interface complète pour uploader et analyser des images en temps réel.
- 🔍 **Explainable AI (XAI)** :
  - **Grad-CAM / Grad-CAM++** : Cartes de chaleur sur les zones d'intérêt.
  - **LIME** : Explication par perturbation locale (superpixels).
- 🛡️ **Optimisation de Sécurité** : Seuils de décision ajustés pour maximiser le rappel sur les cancers (Melanoma).

---

## 🚀 Démarrage Rapide (Quick Start)

### 1. Pré-requis

Assurez-vous d'avoir Python 3.10 ou 3.11 installé. L'utilisation d'un environnement virtuel est recommandée.

```bash
# Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installation

```bash
# Installer les dépendances
pip install -r requirements.txt
```

### 3. Lancer l'Application

C'est la méthode principale pour utiliser le projet.

```bash
python3 -m streamlit run src/app.py
```

L'application s'ouvrira dans votre navigateur (http://localhost:8501).

---

## 🛠️ Pipeline de Données et Entraînement

Si vous souhaitez reproduire l'entraînement complet :

### 1. Préparation des données

Télécharge le dataset HAM10000 et crée un split propre (Train 80% / Val 10% / Test 10%) pour éviter les fuites de données.

```bash
python src/download_data.py
python src/split_data.py
```

### 2. Entraînement

Lance l'entraînement avec Focal Loss et MixUp. Le meilleur modèle sera sauvegardé dans `models/best_model_finetuned.keras`.

```bash
python src/train.py
```

### 3. Optimisation et Évaluation

Génère les rapports de performance et calcule les seuils optimaux pour maximiser la détection des cancers.

```bash
python src/evaluate.py
python src/threshold_optimizer.py
```

## 📁 Structure du Projet

```
.
├── data/                       # Données images
│   ├── raw/                    # Raw downloads
│   └── split/                  # Train/Val/Test directories
├── models/
│   ├── best_model_finetuned.keras  # Modèle final
│   └── class_mapping.json      # Index -> Nom de classe
├── results/
│   ├── best_samples.json       # Échantillons de démonstration curés
│   ├── optimized_thresholds.json
│   └── explanations/           # Outputs XAI sauvés
├── src/
│   ├── app.py                  # ➤ APPLICATION PRINCIPALE
│   ├── train.py                # Script d'entraînement
│   ├── model.py                # Architecture EfficientNet
│   ├── explain.py              # Moteur XAI (Grad-CAM, LIME)
│   ├── split_data.py           # Séparation des données
│   └── ...
└── requirements.txt
```

