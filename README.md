# 🎬 CinéVision

> Analyse data-driven des facteurs de succès au cinéma — Projet Bachelor 3 Data & IA, Ynov Lyon

**Application en ligne :** [cinevision.lucienbernand.com](https://cinevision.lucienbernand.com)

CinéVision est une application de **data storytelling** qui analyse plus de **2 900 films sortis entre 1980 et 2024** pour identifier les facteurs de succès au box-office et prédire les performances d'un nouveau projet de film.

---

## ✨ Fonctionnalités

- 📊 **Exploration** : filtres dynamiques (genre, année, budget), scatter interactif, heatmap de corrélations
- 📈 **Tendances** : évolution des budgets/recettes sur 45 ans, saisonnalité, comparaison franchises vs originaux, top réalisateurs
- 🔮 **Prédiction ML** : simulez un film et obtenez sa probabilité de succès + recettes estimées (Random Forest, F1 = 0.83)
- ⚖️ **Comparateur** : 2 films ou 2 réalisateurs côte à côte avec radar charts

---

## 🚀 Démarrage rapide (local)

### Prérequis
- Python 3.11+
- Une clé API TMDB ([gratuit](https://www.themoviedb.org/settings/api))

### Installation

```bash
# Cloner le repo
git clone https://github.com/lucienbnd/cinevision.git
cd cinevision

# Environnement virtuel
python -m venv .venv
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # Linux/Mac

# Dépendances
pip install -r requirements.txt

# Configurer la clé API
cp .env.example .env
# → éditer .env et y mettre votre clé TMDB
```

### Lancer l'application

```bash
streamlit run "app/🏠_Accueil.py"
```

Ouvrir [http://localhost:8501](http://localhost:8501)

---

## 🐳 Démarrage rapide (Docker)

```bash
docker compose up --build -d
```

L'app sera accessible sur [http://localhost:8501](http://localhost:8501)

---

## 📓 Reproduire le pipeline complet

Les 4 notebooks documentent la démarche :

```bash
jupyter notebook
```

| Notebook | Description |
|---|---|
| `01_data_acquisition.ipynb` | Collecte de ~4 000 films via API TMDB |
| `02_data_cleaning.ipynb` | Nettoyage + feature engineering (37 → 56 colonnes) |
| `03_eda.ipynb` | Analyse exploratoire complète |
| `04_modeling.ipynb` | Entraînement ML (classification + régression) |

Ou via scripts CLI :

```bash
python src/data_acquisition.py --pages 200
python src/data_cleaning.py
python src/feature_engineering.py
python src/model.py
```

---

## 🛠️ Stack technique

| Domaine | Outils |
|---|---|
| Langage | Python 3.13 |
| Données | API TMDB |
| Analyse | pandas, numpy, matplotlib, seaborn |
| ML | scikit-learn, XGBoost |
| Visualisation | Plotly |
| Application | Streamlit |
| Déploiement | Docker, Apache (reverse proxy), Let's Encrypt |
| CI/CD | GitHub Actions |
| Versionning | Git, GitHub |

---

## 📁 Structure du projet

```
cinevision/
├── app/                      # Application Streamlit
│   ├── 🏠_Accueil.py         # Page d'accueil (entrée Streamlit)
│   ├── pages/                # Pages du dashboard
│   └── components/           # Composants graphiques réutilisables (Plotly)
├── src/                      # Pipeline Python
│   ├── data_acquisition.py   # Collecte TMDB
│   ├── data_cleaning.py      # Nettoyage
│   ├── feature_engineering.py# Features ML
│   └── model.py              # Entraînement et inférence
├── notebooks/                # Notebooks Jupyter (rapport)
├── data/
│   ├── raw/                  # Données brutes (non versionnées)
│   └── processed/            # CSV nettoyé (versionné)
├── models/                   # Modèles ML entraînés (.pkl)
├── docs/                     # Documentation technique
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 📊 Performances des modèles ML

| Tâche | Meilleur modèle | Métrique | Valeur |
|---|---|---|---|
| Classification (succès / échec) | Random Forest | F1 | **0.83** |
| Régression (recettes) | Random Forest | R² | **0.51** |

---

## 👤 Auteur

**Lucien Bernand** — Bachelor 3 Data & IA, Ynov Lyon (promo 2025-2026)
