# Projet d'analyse des accidents routiers

Ce projet permet d'analyser les données d'accidents corporels de la circulation en France.

## Prérequis

- Python 3.8+
- pip

## Installation des dépendances

```bash
pip install -r requirements.txt
```

## Exécution du pipeline complet

```bash
python main.py --step all --years 2023
```

### Options disponibles :
- `--step` : 
  - `preprocess` : Prétraitement des données
  - `exploratory` : Analyse exploratoire
  - `ml` : Modélisation machine learning
  - `all` : Toutes les étapes (par défaut)
- `--years` : Années à analyser (ex: 2023 ou 2022,2023)
- `--balance_method` : Méthode de gestion du déséquilibre des classes (`smote`, `undersample`, `smoteenn`)

## Structure des dossiers

- `data/` : Contient les données brutes et nettoyées
- `logs/` : Fichiers de logs d'exécution
- `outputs/` : Résultats d'analyse et modèles entraînés
- `src/` : Code source (prétraitement, analyse, modélisation)

## Fichiers importants

- `main.py` : Point d'entrée principal
- `data_preprocessing.py` : Prétraitement des données
- `exploratory_analysis.py` : Analyse exploratoire
- `ml_modeling.py` : Modélisation machine learning