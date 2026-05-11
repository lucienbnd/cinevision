# Contexte projet — CinéVision : Backlog & Gantt

## Objectif de cette demande

Créer un **backlog complet du projet** sous forme de tâches (type système de tickets / user stories) et un **diagramme de Gantt** qui couvre tous les aspects du projet : cadrage, fonctionnalités et maquettes.

---

## Présentation du projet

**Nom** : CinéVision  
**Type** : Projet académique Bachelor 3 Data & IA — Ynov Lyon  
**Effectif** : 1 personne (solo)  
**Durée estimée** : ~8 semaines  
**Date de début** : 20 avril 2026  
**Orale intermédiaire** : semaine 4 (mi-mai 2026)  
**Rendu final + oral** : semaine 8 (mi-juin 2026)

### Description

Application de data storytelling analysant les facteurs de succès au cinéma et proposant des prédictions de performances au box-office. Le projet couvre l'intégralité du pipeline data science : acquisition de données (API TMDB), nettoyage, analyse exploratoire, modélisation ML et application interactive Streamlit.

### Stack technique

- Python 3.11+
- API TMDB (The Movie Database)
- pandas, numpy, matplotlib, seaborn, plotly
- scikit-learn, XGBoost
- Streamlit (application interactive)
- Git / GitHub

---

## Livrables attendus (contraintes académiques)

1. Dépôt Git avec tout le code et la documentation
2. Jupyter Notebooks retraçant la démarche et les analyses
3. Application de data storytelling déployée localement (Streamlit)
4. Documentation technique du projet et manuel d'installation

---

## Architecture du projet

```
cinevision/
├── data/raw/                 # Données brutes API
├── data/processed/           # Données nettoyées
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── data_acquisition.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── model.py
├── app/
│   ├── app.py                # Point d'entrée Streamlit
│   ├── pages/
│   │   ├── 01_exploration.py
│   │   ├── 02_tendances.py
│   │   ├── 03_prediction.py
│   │   └── 04_comparateur.py
│   └── components/charts.py
├── models/best_model.pkl
└── docs/
```

---

## Phases du projet (à utiliser pour structurer le backlog et le Gantt)

### Phase 1 — Cadrage (semaine 1)

- Définition du sujet et validation professeur
- Rédaction du brief projet
- Étude de faisabilité (exploration API TMDB, limites, volume de données)
- Choix de la stack technique
- Mise en place du repo GitHub
- Création de la structure de dossiers
- Configuration de l'environnement (.env, requirements.txt, .gitignore)
- Planification du backlog et répartition des tâches dans le temps
- Création du Kanban / outil de gestion de projet

### Phase 2 — Acquisition et préparation des données (semaines 2-3)

- Création du compte TMDB et obtention de la clé API
- Développement du script d'acquisition (`src/data_acquisition.py`)
  - Requêtes paginées sur `/discover/movie` (~10 000 films)
  - Détails films via `/movie/{id}` (budget, recettes, runtime, etc.)
  - Credits via `/movie/{id}/credits` (casting, réalisateur)
  - Gestion du rate limiting (40 req/10s)
  - Sauvegarde en CSV
- Notebook `01_data_acquisition.ipynb` documentant le processus
- Développement du pipeline de nettoyage (`src/data_cleaning.py`)
  - Suppression films sans budget/recettes
  - Parsing des genres (multi-label)
  - Extraction réalisateur + top 5 acteurs
  - Création variables dérivées : ROI, is_success, release_month, is_franchise, director_avg_rating, lead_actor_popularity
- Notebook `02_data_cleaning.ipynb`
- Feature engineering (`src/feature_engineering.py`)
- Export du dataset final : `data/processed/movies_clean.csv`

### Phase 3 — Analyse exploratoire (semaine 3-4)

- Notebook `03_eda.ipynb` avec les analyses suivantes :
  - Distribution des budgets et recettes (histogrammes, boxplots)
  - Matrice de corrélation (budget, recettes, notes, popularité)
  - Scatter plot budget vs recettes
  - Top genres par décennie (barplots empilés)
  - Saisonnalité : recettes moyennes par mois de sortie
  - Évolution du nombre de films et budgets moyens par année
  - Analyse franchises vs films originaux
  - Top réalisateurs et acteurs par ROI
  - Statistiques descriptives complètes
- Synthèse des insights clés (pour le storytelling)

### Phase 4 — Modélisation ML (semaines 4-5)

- Préparation des features (encoding, scaling, train/test split)
- **Classification** (prédire succès/échec) :
  - Logistic Regression (baseline)
  - Random Forest Classifier
  - XGBoost Classifier
  - Métriques : accuracy, precision, recall, F1, matrice de confusion
- **Régression** (prédire les recettes) :
  - Linear Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor
  - Métriques : RMSE, MAE, R²
- Validation croisée (5-fold)
- Feature importance analysis
- Hyperparameter tuning (GridSearch)
- Sélection et export du meilleur modèle (`models/best_model.pkl`)
- Notebook `04_modeling.ipynb`
- Script réutilisable `src/model.py`

### Phase 5 — Maquettes et design de l'application (semaine 5)

- Maquettes wireframe des 4 pages :
  - Page d'accueil / introduction storytelling
  - Page Exploration (filtres + graphiques interactifs)
  - Page Tendances (timeline, saisonnalité, évolution genres)
  - Page Prédiction / Simulateur (formulaire + résultat ML)
  - Page Comparateur (2 films ou 2 réalisateurs côte à côte)
- Définition de la charte graphique (couleurs, typographie)
- UX : parcours utilisateur et navigation entre pages
- Choix des composants Streamlit à utiliser par page

### Phase 6 — Développement de l'application Streamlit (semaines 5-7)

- `app/app.py` : page d'accueil avec storytelling introductif
- `app/pages/01_exploration.py` :
  - Filtres interactifs (genre, année, budget min/max)
  - Scatter plot budget/recettes (plotly)
  - Distribution des notes
  - Tableau de données filtrable
- `app/pages/02_tendances.py` :
  - Timeline du cinéma par décennie
  - Analyse saisonnière interactive
  - Évolution des genres dans le temps
- `app/pages/03_prediction.py` :
  - Formulaire : budget, genre, mois de sortie, runtime
  - Chargement du modèle ML
  - Affichage prédiction (recettes estimées + probabilité succès)
  - Jauge visuelle du potentiel
- `app/pages/04_comparateur.py` :
  - Sélection de 2 films ou 2 réalisateurs
  - Radar chart comparatif
  - Statistiques côte à côte
- `app/components/charts.py` : fonctions de visualisation réutilisables
- Tests manuels de chaque page
- Corrections de bugs et polish UI

### Phase 7 — Documentation et finalisation (semaine 7-8)

- README.md complet (installation, configuration, utilisation)
- Documentation technique (`docs/documentation_technique.md`)
  - Architecture du projet
  - Choix techniques justifiés
  - Pipeline de données
  - Résultats ML et interprétation
  - Guide d'utilisation de l'application
- Nettoyage du code (commentaires, docstrings)
- Vérification que les notebooks tournent de A à Z
- Commit final propre
- Préparation de la soutenance orale

### Phase 8 — Préparation oral final (semaine 8)

- Création du support de présentation (slides)
- Structuration du storytelling pour l'oral (15 min)
- Préparation de la démo live de l'application
- Anticipation des questions du jury
- Répétition

---

## Critères d'évaluation (pour prioriser les tâches)

| Critère | Pondération |
|---------|-------------|
| Acquérir des données | 4 |
| Préparer et nettoyer des données | 3 |
| Explorer et analyser des données | 3 |
| Visualiser des données | 3 |
| Appliquer un modèle ML et l'évaluer | 4 |
| Prédire et donner des recommandations | 4 |
| Concevoir et déployer une interface interactive | 2 |
| Documenter son projet | 2 |

**Priorité maximale** : acquisition données + ML + prédictions (pondération 4)  
**Priorité haute** : nettoyage + EDA + visualisation (pondération 3)  
**Priorité moyenne** : app interactive + documentation (pondération 2)

---

## Consignes pour la création du backlog

- Utiliser un format de type **user stories** ou **tâches techniques** numérotées
- Chaque tâche doit avoir : un ID, un titre, une description courte, une priorité (P0/P1/P2/P3), une estimation en jours, une phase, et un statut
- Regrouper les tâches par **Epics** correspondant aux phases ci-dessus
- Les dépendances entre tâches doivent être explicites

## Consignes pour le diagramme de Gantt

- Format : diagramme de Gantt visuel (Mermaid, ou tableau semaine par semaine)
- Couvrir les 8 semaines du projet (20 avril → mi-juin 2026)
- Montrer les dépendances entre phases
- Identifier le chemin critique
- Marquer les jalons : oral intermédiaire (semaine 4), rendu final (semaine 8)
- Prendre en compte qu'une seule personne travaille sur le projet (pas de parallélisme illimité)
- Estimer ~3-4h de travail par jour sur le projet (en parallèle des autres cours)
