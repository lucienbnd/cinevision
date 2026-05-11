"""
Script d'acquisition de données depuis l'API TMDB (The Movie Database).

Ce module collecte des films via l'API TMDB en plusieurs étapes :
1. Découverte de films (endpoint /discover/movie)
2. Récupération des détails de chaque film (endpoint /movie/{id})
3. Récupération des credits (endpoint /movie/{id}/credits)

Les résultats sont sauvegardés en CSV dans data/raw/.

Usage:
    python src/data_acquisition.py --pages 50 --start-year 2000 --end-year 2024
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constantes API TMDB
TMDB_BASE_URL = "https://api.themoviedb.org/3"
RATE_LIMIT_SLEEP = 0.25  # ~4 req/s, sous la limite de 40 req/10s
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # secondes

# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def get_api_key() -> str:
    """Charge la clé API TMDB depuis le fichier .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise ValueError(
            "TMDB_API_KEY non trouvée. Vérifiez votre fichier .env."
        )
    return api_key


def make_request(
    url: str, params: dict[str, Any], retries: int = MAX_RETRIES
) -> dict[str, Any] | None:
    """
    Effectue une requête HTTP avec gestion des erreurs et retry exponentiel.

    Args:
        url: URL de l'endpoint TMDB
        params: paramètres de la requête (inclut api_key)
        retries: nombre de tentatives en cas d'échec

    Returns:
        Le JSON de la réponse, ou None en cas d'échec final.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)

            # Rate limit dépassé -> attendre
            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 10))
                logger.warning(f"Rate limit atteint, attente {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            wait_time = RETRY_BACKOFF**attempt
            logger.warning(
                f"Erreur requête (tentative {attempt + 1}/{retries}): {e}. "
                f"Nouvelle tentative dans {wait_time}s..."
            )
            time.sleep(wait_time)

    logger.error(f"Échec définitif de la requête : {url}")
    return None


def discover_movies(
    api_key: str,
    pages: int = 50,
    start_year: int = 2000,
    end_year: int = 2024,
    min_votes: int = 50,
) -> list[dict[str, Any]]:
    """
    Récupère une liste de films via l'endpoint /discover/movie.

    Args:
        api_key: clé API TMDB
        pages: nombre de pages à récupérer (20 films par page)
        start_year: année minimum de sortie
        end_year: année maximum de sortie
        min_votes: nombre minimum de votes (filtre la qualité)

    Returns:
        Liste de dictionnaires de films (résultats partiels).
    """
    logger.info(
        f"Découverte de films ({start_year}-{end_year}, "
        f"min {min_votes} votes, {pages} pages)..."
    )
    movies = []
    url = f"{TMDB_BASE_URL}/discover/movie"

    for page in range(1, pages + 1):
        params = {
            "api_key": api_key,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "include_video": "false",
            "page": page,
            "primary_release_date.gte": f"{start_year}-01-01",
            "primary_release_date.lte": f"{end_year}-12-31",
            "vote_count.gte": min_votes,
        }

        data = make_request(url, params)
        if data is None:
            logger.warning(f"Page {page} ignorée (erreur API)")
            continue

        results = data.get("results", [])
        movies.extend(results)

        if page % 10 == 0:
            logger.info(f"  Page {page}/{pages} traitée ({len(movies)} films)")

        time.sleep(RATE_LIMIT_SLEEP)

    logger.info(f"Découverte terminée : {len(movies)} films collectés")
    return movies


def fetch_movie_details(api_key: str, movie_id: int) -> dict[str, Any] | None:
    """Récupère les détails complets d'un film (budget, recettes, runtime...)."""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = {"api_key": api_key, "language": "en-US"}
    return make_request(url, params)


def fetch_movie_credits(api_key: str, movie_id: int) -> dict[str, Any] | None:
    """Récupère le casting et l'équipe technique d'un film."""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}/credits"
    params = {"api_key": api_key}
    return make_request(url, params)


def enrich_movies(
    api_key: str, movies: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Enrichit chaque film avec ses détails complets et ses credits.

    Pour chaque film :
    - Appel /movie/{id} pour budget, revenue, runtime, etc.
    - Appel /movie/{id}/credits pour casting et réalisateur

    Args:
        api_key: clé API TMDB
        movies: liste de films issue de discover_movies()

    Returns:
        Liste de films enrichis (fusion discover + details + credits).
    """
    logger.info(f"Enrichissement de {len(movies)} films...")
    enriched = []

    for idx, movie in enumerate(movies, start=1):
        movie_id = movie.get("id")
        if not movie_id:
            continue

        # Récupération des détails
        details = fetch_movie_details(api_key, movie_id)
        time.sleep(RATE_LIMIT_SLEEP)

        # Récupération des credits
        credits = fetch_movie_credits(api_key, movie_id)
        time.sleep(RATE_LIMIT_SLEEP)

        if details is None or credits is None:
            continue

        # Extraction des champs utiles
        cast = credits.get("cast", [])[:5]  # top 5 acteurs
        crew = credits.get("crew", [])
        director = next(
            (c["name"] for c in crew if c.get("job") == "Director"), None
        )
        composer = next(
            (c["name"] for c in crew if c.get("job") == "Original Music Composer"),
            None,
        )

        enriched_movie = {
            "id": movie_id,
            "title": details.get("title"),
            "original_title": details.get("original_title"),
            "release_date": details.get("release_date"),
            "runtime": details.get("runtime"),
            "budget": details.get("budget"),
            "revenue": details.get("revenue"),
            "vote_average": details.get("vote_average"),
            "vote_count": details.get("vote_count"),
            "popularity": details.get("popularity"),
            "original_language": details.get("original_language"),
            "status": details.get("status"),
            "genres": [g["name"] for g in details.get("genres", [])],
            "production_companies": [
                p["name"] for p in details.get("production_companies", [])
            ],
            "production_countries": [
                c["iso_3166_1"] for c in details.get("production_countries", [])
            ],
            "belongs_to_collection": details.get("belongs_to_collection") is not None,
            "collection_name": (
                details.get("belongs_to_collection", {}).get("name")
                if details.get("belongs_to_collection")
                else None
            ),
            "director": director,
            "composer": composer,
            "cast_top5": [c["name"] for c in cast],
            "cast_top5_popularity": [c.get("popularity", 0) for c in cast],
            "overview": details.get("overview"),
            "tagline": details.get("tagline"),
        }

        enriched.append(enriched_movie)

        if idx % 50 == 0:
            logger.info(f"  {idx}/{len(movies)} films enrichis")

    logger.info(f"Enrichissement terminé : {len(enriched)} films")
    return enriched


def save_to_csv(movies: list[dict[str, Any]], filename: str) -> Path:
    """Sauvegarde la liste de films dans un fichier CSV."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_RAW_DIR / filename
    df = pd.DataFrame(movies)
    df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info(f"Données sauvegardées : {filepath} ({len(df)} lignes)")
    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquisition de données TMDB")
    parser.add_argument(
        "--pages", type=int, default=50, help="Nombre de pages à récupérer (20 films/page)"
    )
    parser.add_argument(
        "--start-year", type=int, default=2000, help="Année minimum"
    )
    parser.add_argument(
        "--end-year", type=int, default=2024, help="Année maximum"
    )
    parser.add_argument(
        "--min-votes", type=int, default=50, help="Nombre minimum de votes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="movies_raw.csv",
        help="Nom du fichier de sortie",
    )
    args = parser.parse_args()

    api_key = get_api_key()

    # Étape 1 : découverte
    movies = discover_movies(
        api_key,
        pages=args.pages,
        start_year=args.start_year,
        end_year=args.end_year,
        min_votes=args.min_votes,
    )

    # Étape 2 : enrichissement
    enriched = enrich_movies(api_key, movies)

    # Étape 3 : sauvegarde
    save_to_csv(enriched, args.output)


if __name__ == "__main__":
    main()
