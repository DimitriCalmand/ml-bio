# Classification des Maladies de Peau - HAM10000

Application de classification automatique des maladies de peau utilisant le dataset HAM10000 avec des fonctionnalités d'interprétabilité (XAI).

## Description

Ce projet utilise le deep learning (Transfer Learning avec MobileNetV2) pour classifier 7 types de lésions cutanées :

| Code | Maladie | Type |
|------|---------|------|
| **mel** | Melanoma | 🔴 Cancéreux |
| **bcc** | Basal Cell Carcinoma | 🔴 Cancéreux |
| **akiec** | Actinic Keratosis | 🟠 Pré-cancéreux |
| **bkl** | Benign Keratosis | 🟢 Bénin |
| **nv** | Melanocytic Nevus | 🟢 Bénin |
| **df** | Dermatofibroma | 🟢 Bénin |
| **vasc** | Vascular Lesion | 🟢 Bénin |

## Fonctionnalités

- ✅ **Classification automatique** avec MobileNetV2 (Transfer Learning)
- ✅ **Fine-tuning** en deux phases pour de meilleures performances
- ✅ **Gestion du déséquilibre** des classes avec class weights
- ✅ **Augmentation de données** (rotation, flip, zoom, etc.)
- ✅ **Interprétabilité avec Grad-CAM** - Visualisation des régions d'attention
- ✅ **Interprétabilité avec Grad-CAM++** - Amélioration de Grad-CAM
- ✅ **Interprétabilité avec LIME** - Explications locales
- ✅ **Rapports cliniques** - Visualisations adaptées au contexte médical
- ✅ **Optimisation des seuils** pour classes critiques (mélanome)

## Installation

```bash
# Cloner le projet
git clone <repository-url>
cd ml-bio

# Installer les dépendances
pip install -r requirements.txt
```

## Configuration Kaggle

Pour télécharger le dataset, configurez votre API Kaggle :

1. Créez un compte sur [Kaggle](https://www.kaggle.com/)
2. Allez dans Account > API > Create New API Token
3. Placez le fichier `kaggle.json` dans `~/.kaggle/`
4. Définissez les permissions : `chmod 600 ~/.kaggle/kaggle.json`

## Utilisation

### 1. Télécharger et préparer les données

```bash
python src/download_data.py
```

### 2. Entraîner le modèle

```bash
python src/train.py
```

### 3. Évaluer le modèle

```bash
python src/evaluate.py
```

### 4. Générer des explications (XAI)

```bash
# Explication basique (Grad-CAM + LIME)
python src/generate_explanation.py path/to/image.jpg

# Rapport clinique complet
python src/generate_explanation.py path/to/image.jpg --clinical

# Toutes les méthodes d'explication
python src/generate_explanation.py path/to/image.jpg --methods gradcam gradcam++ lime

# Traitement par lot
python src/generate_explanation.py path/to/folder/ --batch

# LIME haute qualité (plus d'échantillons)
python src/generate_explanation.py path/to/image.jpg --lime-samples 2000
```

### 5. Tester les fonctionnalités d'explication

```bash
# Test rapide sur une image aléatoire
python src/test_explain.py

# Test sur une image spécifique
python src/test_explain.py --image path/to/image.jpg

# Test sur toutes les classes
python src/test_explain.py --all
```

## Structure du Projet

```
.
├── data/
│   ├── raw/                    # Données brutes HAM10000
│   └── processed/              # Images organisées par classe
├── models/
│   ├── best_model_finetuned.keras  # Meilleur modèle (fine-tuned)
│   ├── best_model.keras        # Modèle phase 1
│   ├── class_mapping.json      # Mapping des classes
│   └── training_config.json    # Configuration d'entraînement
├── results/
│   ├── explanations/           # Visualisations XAI générées
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── evaluation_results.json
├── src/
│   ├── download_data.py        # Téléchargement des données
│   ├── model.py                # Architecture du modèle
│   ├── train.py                # Entraînement (2 phases)
│   ├── evaluate.py             # Évaluation et métriques
│   ├── explain.py              # Module XAI (Grad-CAM, LIME)
│   ├── generate_explanation.py # CLI pour les explications
│   ├── test_explain.py         # Tests des explications
│   └── threshold_optimizer.py  # Optimisation des seuils
├── requirements.txt
└── README.md
```

## Interprétabilité (XAI)

### Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM visualise les régions de l'image qui ont le plus contribué à la décision du modèle. Les zones en rouge/jaune indiquent une forte attention du modèle.

**Référence**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.

### Grad-CAM++

Version améliorée de Grad-CAM avec une meilleure localisation, particulièrement utile quand plusieurs instances du même type de lésion sont présentes.

**Référence**: Chattopadhyay et al., "Grad-CAM++: Generalized Gradient-based Visual Explanations", WACV 2018.

### LIME (Local Interpretable Model-agnostic Explanations)

LIME identifie les superpixels (régions de l'image) qui influencent positivement ou négativement la prédiction. C'est une méthode model-agnostic.

**Référence**: Ribeiro et al., "Why Should I Trust You?: Explaining the Predictions of Any Classifier", KDD 2016.

## Exemple d'utilisation Python

```python
from tensorflow import keras
from src.explain import ExplanationGenerator, create_clinical_explanation
import json

# Charger le modèle
model = keras.models.load_model("models/best_model_finetuned.keras")

# Charger les noms de classes
with open("models/class_mapping.json") as f:
    class_mapping = json.load(f)
class_names = [class_mapping[str(i)] for i in range(7)]

# Générer une explication complète
generator = ExplanationGenerator(model, class_names)
result = generator.explain_image(
    "path/to/image.jpg",
    methods=['gradcam', 'gradcam++', 'lime']
)

# Créer la visualisation
generator.create_explanation_figure(result, save_path="explanation.png")

# Ou générer un rapport clinique
create_clinical_explanation(
    model, "path/to/image.jpg", class_names,
    output_path="clinical_report.png"
)
```

## Résultats

Les résultats d'entraînement et d'évaluation sont sauvegardés dans `results/`:
- `confusion_matrix.png` - Matrice de confusion
- `roc_curves.png` - Courbes ROC par classe
- `classification_report.txt` - Rapport détaillé
- `evaluation_results.json` - Métriques JSON
- `explanations/` - Visualisations XAI

## Avertissement

⚠️ **AVERTISSEMENT MÉDICAL**: Cet outil est une aide à la décision et ne remplace pas l'expertise d'un dermatologue. Tout diagnostic doit être confirmé par un examen clinique et une analyse histopathologique par un professionnel de santé qualifié.
