# Classification des Maladies de Peau - HAM10000

Application de classification automatique des maladies de peau utilisant le dataset HAM10000.

## Description

Ce projet utilise le deep learning pour classifier 7 types de lésions cutanées :
- Melanoma (mel)
- Melanocytic nevus (nv)
- Basal cell carcinoma (bcc)
- Actinic keratosis (akiec)
- Benign keratosis (bk)
- Dermatofibroma (df)
- Vascular lesion (vasc)

## Installation

```bash
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

## Structure du Projet

```
.
├── data/                   # Données (créé automatiquement)
├── models/                 # Modèles entraînés (créé automatiquement)
├── results/                # Résultats et graphiques (créé automatiquement)
├── src/
│   ├── download_data.py   # Téléchargement des données
│   ├── model.py           # Définition du modèle
│   ├── train.py           # Entraînement
│   └── evaluate.py        # Évaluation
├── requirements.txt
└── README.md
```

## Résultats

Les résultats d'entraînement et d'évaluation seront sauvegardés dans le dossier `results/`.
