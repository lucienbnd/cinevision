"""
Pipeline de nettoyage et préparation des données TMDB.

Ce module transforme les données brutes (data/raw/movies_raw.csv) en un dataset
exploitable pour l'analyse exploratoire et la modélisation ML.

Étapes :
1. Chargement et parsing des colonnes complexes (listes stockées en string)
2. Filtrage des films sans budget ni recettes
3. Gestion des valeurs manquantes
4. Suppression des doublons
5. Détection et traitement des valeurs aberrantes
6. Sortie : data/processed/movies_clean.csv
"""

import ast
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_list_column(value):
    """
    Convertit une chaîne représentant une liste Python en vraie liste.

    Les colonnes comme genres, cast_top5, production_companies sont stockées
    en CSV comme des strings de la forme "['Action', 'Drama']".
    """
    if pd.isna(value) or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def load_raw_data(filename: str = "movies_raw.csv") -> pd.DataFrame:
    """Charge le CSV brut et parse les colonnes complexes."""
    filepath = DATA_RAW_DIR / filename
    logger.info(f"Chargement de {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset brut : {df.shape[0]} films, {df.shape[1]} colonnes")

    # Parser les colonnes contenant des listes
    list_cols = [
        "genres",
        "production_companies",
        "production_countries",
        "cast_top5",
        "cast_top5_popularity",
    ]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_column)

    return df


def filter_invalid_movies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre les films inexploitables.

    Critères :
    - Budget ET recettes > 0 (nécessaire pour calculer ROI et faire du ML)
    - Date de sortie valide
    - Au moins un genre renseigné
    """
    initial = len(df)
    logger.info("Filtrage des films invalides...")

    # Budget et recettes > 0
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    logger.info(f"  Après filtre budget+revenue > 0 : {len(df)} films (-{initial - len(df)})")

    # Date de sortie valide
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["release_date"]).copy()
    logger.info(f"  Après filtre date valide : {len(df)} films (-{before - len(df)})")

    # Au moins un genre
    before = len(df)
    df = df[df["genres"].apply(lambda g: len(g) > 0)].copy()
    logger.info(f"  Après filtre genre non vide : {len(df)} films (-{before - len(df)})")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons basés sur l'ID TMDB."""
    initial = len(df)
    df = df.drop_duplicates(subset=["id"]).copy()
    logger.info(f"Suppression doublons : {initial - len(df)} doublons supprimés → {len(df)} films")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traite les valeurs aberrantes.

    - Budget < 1000 $ = probablement erroné (films avec budget trop faible suspect)
    - Runtime < 40 min ou > 240 min = probablement erroné ou court-métrage
    """
    initial = len(df)
    logger.info("Traitement des valeurs aberrantes...")

    df = df[df["budget"] >= 1000].copy()
    logger.info(f"  Après filtre budget >= 1000$ : {len(df)} films")

    df = df[(df["runtime"] >= 40) & (df["runtime"] <= 240)].copy()
    logger.info(f"  Après filtre runtime (40-240 min) : {len(df)} films")

    logger.info(f"Total films retenus : {len(df)} (-{initial - len(df)})")
    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les variables dérivées de base (utilisables sans calculs lourds).

    Note : les features avancées (popularité acteur, ROI moyen réalisateur)
    sont calculées dans feature_engineering.py.
    """
    logger.info("Création des features de base...")

    # ROI = (recettes - budget) / budget
    df["roi"] = (df["revenue"] - df["budget"]) / df["budget"]

    # Succès = ROI > 1 (le film a doublé sa mise)
    df["is_success"] = (df["roi"] > 1).astype(int)

    # Features temporelles
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_day_of_week"] = df["release_date"].dt.dayofweek  # 0=lundi
    df["release_decade"] = (df["release_year"] // 10) * 10

    # Saison de sortie
    season_map = {
        12: "Hiver", 1: "Hiver", 2: "Hiver",
        3: "Printemps", 4: "Printemps", 5: "Printemps",
        6: "Été", 7: "Été", 8: "Été",
        9: "Automne", 10: "Automne", 11: "Automne",
    }
    df["release_season"] = df["release_month"].map(season_map)

    # Genre principal (le premier listé)
    df["main_genre"] = df["genres"].apply(lambda g: g[0] if len(g) > 0 else "Unknown")
    df["nb_genres"] = df["genres"].apply(len)

    # Popularité moyenne du top casting
    df["cast_avg_popularity"] = df["cast_top5_popularity"].apply(
        lambda lst: sum(lst) / len(lst) if len(lst) > 0 else 0
    )
    df["lead_actor_popularity"] = df["cast_top5_popularity"].apply(
        lambda lst: lst[0] if len(lst) > 0 else 0
    )

    # Nombre de sociétés de production
    df["nb_production_companies"] = df["production_companies"].apply(len)

    # Film international (plusieurs pays de production)
    df["is_international"] = (
        df["production_countries"].apply(len) > 1
    ).astype(int)

    # Franchise → déjà existant : belongs_to_collection
    df["is_franchise"] = df["belongs_to_collection"].astype(int)

    logger.info(f"Features de base créées : {df.shape[1]} colonnes au total")
    return df


def save_clean_data(df: pd.DataFrame, filename: str = "movies_clean.csv") -> Path:
    """Sauvegarde le dataset nettoyé."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_PROCESSED_DIR / filename

    # Convertir les listes en strings pour CSV
    df_to_save = df.copy()
    list_cols = ["genres", "production_companies", "production_countries",
                 "cast_top5", "cast_top5_popularity"]
    for col in list_cols:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].apply(str)

    df_to_save.to_csv(filepath, index=False, encoding="utf-8")
    logger.info(f"Données nettoyées sauvegardées : {filepath} ({len(df_to_save)} lignes)")
    return filepath


def clean_pipeline(input_file: str = "movies_raw.csv",
                   output_file: str = "movies_clean.csv") -> pd.DataFrame:
    """
    Pipeline complet de nettoyage.

    Args:
        input_file: nom du CSV brut
        output_file: nom du CSV nettoyé

    Returns:
        DataFrame nettoyé.
    """
    df = load_raw_data(input_file)
    df = filter_invalid_movies(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = add_basic_features(df)
    save_clean_data(df, output_file)
    return df


if __name__ == "__main__":
    df = clean_pipeline()
    print("\nAperçu du dataset nettoyé :")
    print(df.head())
    print(f"\nShape : {df.shape}")
    print(f"Colonnes : {df.columns.tolist()}")
