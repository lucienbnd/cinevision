"""Page Exploration — explorer interactivement le catalogue de films."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from components.charts import (
    budget_revenue_scatter,
    correlation_heatmap,
    get_all_genres,
    load_movies,
)

st.set_page_config(page_title="Exploration", page_icon="📊", layout="wide")

df = load_movies()

st.title("📊 Exploration des données")
st.markdown(
    "Filtrez et explorez les **2 976 films** du dataset selon vos critères. "
    "Toutes les visualisations sont interactives (hover, zoom)."
)
st.markdown("---")

# ===== Sidebar : filtres =====
st.sidebar.header("🎛️ Filtres")

genres_all = get_all_genres(df)
selected_genres = st.sidebar.multiselect(
    "Genres", genres_all, default=[],
    help="Laissez vide pour tout afficher",
)

year_min, year_max = int(df["release_year"].min()), int(df["release_year"].max())
year_range = st.sidebar.slider(
    "Année de sortie", year_min, year_max, (year_min, year_max),
)

budget_max = float(df["budget"].max())
budget_range = st.sidebar.slider(
    "Budget (millions $)",
    0.0, budget_max / 1e6,
    (0.0, budget_max / 1e6),
)

min_vote = st.sidebar.slider("Note TMDB minimum", 0.0, 10.0, 0.0, 0.5)

franchise_filter = st.sidebar.radio(
    "Type de film", ["Tous", "Franchises uniquement", "Films originaux uniquement"],
)

# ===== Application des filtres =====
mask = (
    (df["release_year"] >= year_range[0])
    & (df["release_year"] <= year_range[1])
    & (df["budget"] >= budget_range[0] * 1e6)
    & (df["budget"] <= budget_range[1] * 1e6)
    & (df["vote_average"] >= min_vote)
)

if selected_genres:
    mask &= df["genres"].apply(lambda g: any(s in g for s in selected_genres))

if franchise_filter == "Franchises uniquement":
    mask &= df["is_franchise"] == 1
elif franchise_filter == "Films originaux uniquement":
    mask &= df["is_franchise"] == 0

df_filtered = df[mask].copy()

# ===== KPIs filtrés =====
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Films", f"{len(df_filtered):,}")
c2.metric("Budget médian", f"${df_filtered['budget'].median() / 1e6:.1f} M")
c3.metric("Recettes médianes", f"${df_filtered['revenue'].median() / 1e6:.1f} M")
c4.metric("Note moyenne", f"{df_filtered['vote_average'].mean():.2f} / 10")
c5.metric("Taux de succès", f"{df_filtered['is_success'].mean()*100:.1f}%")

st.markdown("---")

if len(df_filtered) == 0:
    st.warning("⚠️ Aucun film ne correspond à vos filtres. Élargissez la sélection.")
else:
    # ===== Graphique principal =====
    st.plotly_chart(budget_revenue_scatter(df_filtered), use_container_width=True)

    # ===== Tableau interactif =====
    st.subheader("📋 Liste des films filtrés")

    cols_to_show = [
        "title", "release_year", "main_genre", "director", "budget",
        "revenue", "roi", "vote_average", "is_franchise",
    ]
    df_display = df_filtered[cols_to_show].copy()
    df_display.columns = ["Titre", "Année", "Genre principal", "Réalisateur",
                          "Budget ($)", "Recettes ($)", "ROI", "Note", "Franchise ?"]
    df_display["Franchise ?"] = df_display["Franchise ?"].map({1: "✅", 0: "—"})

    sort_col = st.selectbox(
        "Trier par",
        ["Recettes ($)", "Budget ($)", "ROI", "Note", "Année"],
        index=0,
    )
    ascending = st.checkbox("Ordre croissant", value=False)
    df_display = df_display.sort_values(sort_col, ascending=ascending)

    st.dataframe(df_display, use_container_width=True, hide_index=True, height=420)

    st.markdown("---")

    # ===== Heatmap de corrélation =====
    with st.expander("🔥 Voir la matrice de corrélation des variables clés"):
        st.plotly_chart(correlation_heatmap(df_filtered), use_container_width=True)
