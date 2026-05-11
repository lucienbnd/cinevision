"""
CinéVision — Application de data storytelling sur le cinéma mondial.

Lancement : streamlit run app/app.py

Cette application présente une analyse interactive des facteurs de succès
au cinéma et permet de simuler le potentiel d'un film via un modèle ML.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from components.charts import load_movies

# Configuration de la page
st.set_page_config(
    page_title="CinéVision",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chargement des données (cache)
df = load_movies()

# ===== Header =====
st.title("🎬 CinéVision")
st.markdown(
    "### *Analyse des facteurs de succès au cinéma*"
)
st.markdown("---")

# ===== KPIs en haut =====
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎞️ Films analysés", f"{len(df):,}")
col2.metric("📅 Période", f"{df['release_year'].min()} — {df['release_year'].max()}")
col3.metric("💰 Budget total", f"${df['budget'].sum() / 1e9:,.1f} Mds")
col4.metric("🏆 Taux de succès", f"{df['is_success'].mean()*100:.1f}%")

st.markdown("---")

# ===== Introduction storytelling =====
st.markdown(
    """
    ## Bienvenue dans CinéVision

    **L'industrie du cinéma brasse des milliards chaque année.** Mais qu'est-ce qui
    sépare un blockbuster d'un flop retentissant ? Le budget ? Le casting ? Le mois
    de sortie ? Le genre ? Ou encore le réalisateur ?

    Cette application analyse **plus de 2 900 films sortis entre 1980 et 2024**
    pour répondre à ces questions à travers la donnée. Au programme :

    - 📊 **Explorer** les données filmographiques et financières
    - 📈 **Décrypter** les tendances du cinéma sur 45 ans
    - 🔮 **Simuler** le potentiel de succès d'un nouveau film grâce à l'IA
    - ⚖️ **Comparer** films, réalisateurs et franchises

    👉 **Naviguez via le menu latéral** pour explorer les différentes analyses.
    """
)

st.markdown("---")

# ===== Aperçu des 4 sections =====
st.subheader("🗂️ Les 4 espaces de l'application")

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
        ### 📊 Exploration
        Plongez dans les données. Filtrez par genre, année, budget…
        Visualisez les relations entre budgets, recettes, notes et popularité.
        """
    )
    st.markdown(
        """
        ### 🔮 Prédiction
        Simulez le potentiel d'un film en configurant ses caractéristiques.
        Le modèle ML prédit ses chances de succès et ses recettes estimées.
        """
    )

with c2:
    st.markdown(
        """
        ### 📈 Tendances
        Comment le cinéma a-t-il évolué depuis 1980 ?
        Saisonnalité, genres dominants, montée des franchises…
        """
    )
    st.markdown(
        """
        ### ⚖️ Comparateur
        Comparez 2 films ou 2 réalisateurs côte à côte.
        Radar charts, statistiques et palmarès.
        """
    )

st.markdown("---")

# ===== Insights clés =====
st.subheader("💡 Quelques chiffres-clés à découvrir")

c1, c2, c3 = st.columns(3)

with c1:
    franchise_rate = df[df["is_franchise"] == 1]["is_success"].mean() * 100
    original_rate = df[df["is_franchise"] == 0]["is_success"].mean() * 100
    st.info(
        f"**Franchises vs Films originaux**\n\n"
        f"Une franchise a {franchise_rate:.0f}% de chances de succès, "
        f"contre {original_rate:.0f}% pour un film original."
    )

with c2:
    summer_rate = df[df["release_month"].isin([5, 6, 7])]["is_success"].mean() * 100
    sept_rate = df[df["release_month"] == 9]["is_success"].mean() * 100
    st.success(
        f"**L'effet été**\n\n"
        f"Sortir en mai/juin/juillet → {summer_rate:.0f}% de succès. "
        f"Sortir en septembre → {sept_rate:.0f}%."
    )

with c3:
    df_g = df.explode("genres")
    horror_roi = df_g[df_g["genres"] == "Horror"]["roi"].median()
    st.warning(
        f"**L'horreur, reine du ROI**\n\n"
        f"ROI médian des films d'horreur : **{horror_roi:.1f}x** le budget initial. "
        f"Le genre le plus rentable proportionnellement."
    )

st.markdown("---")
st.caption(
    "Projet réalisé par Lucien Bernand dans le cadre du cours Projet Fil rouge du Bachelor 3 Data & IA — Ynov Lyon · "
    "Données : The Movie Database (TMDB) · "
    "Réalisé avec Streamlit, Plotly, scikit-learn"
)
