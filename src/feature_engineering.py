"""
Feature engineering avancé pour le dataset CinéVision.

Ce module ajoute des features dérivées plus complexes au dataset nettoyé :
- Historique de performance par réalisateur
- Popularité historique des acteurs
- Encodage des genres (one-hot)
- Features d'inflation (budget en valeur constante)

Sortie : data/processed/movies_features.csv
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
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_list_column(value):
    """Convertit une string représentant une liste en liste Python."""
    if pd.isna(value) or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def load_clean_data(filename: str = "movies_clean.csv") -> pd.DataFrame:
    """Charge le CSV nettoyé et re-parse les colonnes liste."""
    filepath = DATA_PROCESSED_DIR / filename
    logger.info(f"Chargement de {filepath}")
    df = pd.read_csv(filepath)
    list_cols = ["genres", "production_companies", "production_countries",
                 "cast_top5", "cast_top5_popularity"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_column)
    df["release_date"] = pd.to_datetime(df["release_date"])
    return df


def add_director_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque film, calcule l'expérience et la performance moyenne du réalisateur
    sur ses films précédents (uniquement, pour éviter le data leakage).
    """
    logger.info("Calcul des features 'réalisateur'...")
    df = df.sort_values("release_date").copy()

    # Stats cumulatives pour chaque réalisateur (avant le film en cours)
    df["director_nb_movies_prior"] = df.groupby("director").cumcount()

    # Moyenne mobile de la note des films précédents du réalisateur
    df["director_avg_vote_prior"] = (
        df.groupby("director")["vote_average"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # Moyenne mobile du ROI des films précédents du réalisateur
    df["director_avg_roi_prior"] = (
        df.groupby("director")["roi"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # Remplir les NaN (premier film du réalisateur) avec la moyenne globale
    df["director_avg_vote_prior"] = df["director_avg_vote_prior"].fillna(
        df["vote_average"].mean()
    )
    df["director_avg_roi_prior"] = df["director_avg_roi_prior"].fillna(
        df["roi"].median()
    )

    logger.info("  Features réalisateur ajoutées")
    return df


def add_genre_features(df: pd.DataFrame, top_n_genres: int = 12) -> pd.DataFrame:
    """One-hot encoding des principaux genres (multi-label)."""
    logger.info(f"Création one-hot encoding pour les {top_n_genres} genres principaux...")

    # Identifier les genres les plus fréquents
    all_genres = [g for genres in df["genres"] for g in genres]
    genre_counts = pd.Series(all_genres).value_counts()
    top_genres = genre_counts.head(top_n_genres).index.tolist()

    # Créer une colonne binaire par genre
    for genre in top_genres:
        col_name = f"genre_{genre.lower().replace(' ', '_')}"
        df[col_name] = df["genres"].apply(lambda g: int(genre in g))

    logger.info(f"  Genres encodés : {top_genres}")
    return df


def add_inflation_adjusted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute le budget et les recettes ajustés à l'inflation (base 2024).

    Utilise un facteur d'inflation simple basé sur l'IPC US (approximation).
    """
    logger.info("Ajustement à l'inflation (base 2024)...")

    # Facteurs d'inflation approximatifs vs 2024 (IPC US)
    # Source: BLS - simplifié pour le projet
    inflation_factors = {
        1980: 3.85, 1981: 3.49, 1982: 3.29, 1983: 3.19, 1984: 3.06,
        1985: 2.95, 1986: 2.90, 1987: 2.80, 1988: 2.69, 1989: 2.56,
        1990: 2.43, 1991: 2.33, 1992: 2.26, 1993: 2.20, 1994: 2.14,
        1995: 2.08, 1996: 2.02, 1997: 1.98, 1998: 1.95, 1999: 1.91,
        2000: 1.84, 2001: 1.79, 2002: 1.77, 2003: 1.73, 2004: 1.68,
        2005: 1.63, 2006: 1.58, 2007: 1.54, 2008: 1.48, 2009: 1.48,
        2010: 1.46, 2011: 1.41, 2012: 1.38, 2013: 1.36, 2014: 1.34,
        2015: 1.34, 2016: 1.32, 2017: 1.29, 2018: 1.26, 2019: 1.24,
        2020: 1.22, 2021: 1.17, 2022: 1.08, 2023: 1.03, 2024: 1.00,
    }
    df["inflation_factor"] = df["release_year"].map(inflation_factors).fillna(1.0)
    df["budget_2024"] = df["budget"] * df["inflation_factor"]
    df["revenue_2024"] = df["revenue"] * df["inflation_factor"]

    logger.info("  Budget et recettes ajustés à l'inflation 2024")
    return df


def add_collection_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque film de franchise, indique le numéro d'ordre dans la collection."""
    logger.info("Features de franchise...")

    df = df.sort_values(["collection_name", "release_date"]).copy()
    df["franchise_position"] = (
        df.groupby("collection_name").cumcount() + 1
    ).where(df["is_franchise"] == 1, 0)

    return df


def feature_engineering_pipeline(
    input_file: str = "movies_clean.csv",
    output_file: str = "movies_features.csv",
) -> pd.DataFrame:
    """Pipeline complet de feature engineering avancé."""
    df = load_clean_data(input_file)

    df = add_director_features(df)
    df = add_genre_features(df)
    df = add_inflation_adjusted_features(df)
    df = add_collection_features(df)

    # Sauvegarde
    filepath = DATA_PROCESSED_DIR / output_file
    df_to_save = df.copy()
    list_cols = ["genres", "production_companies", "production_countries",
                 "cast_top5", "cast_top5_popularity"]
    for col in list_cols:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].apply(str)
    df_to_save.to_csv(filepath, index=False, encoding="utf-8")
    logger.info(f"Dataset avec features sauvegardé : {filepath} ({len(df_to_save)} lignes)")

    return df


if __name__ == "__main__":
    df = feature_engineering_pipeline()
    print("\nAperçu :")
    print(df.head())
    print(f"\nShape : {df.shape}")
    print(f"Nouvelles colonnes : {[c for c in df.columns if c.startswith('director_') or c.startswith('genre_') or 'inflation' in c or '2024' in c or 'franchise_' in c]}")
