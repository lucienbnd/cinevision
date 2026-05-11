"""Page Tendances — évolutions historiques du cinéma sur 45 ans."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from components.charts import (
    director_top_chart,
    films_timeline,
    franchise_vs_original,
    genre_distribution,
    genre_roi_comparison,
    load_movies,
    revenue_by_decade,
    success_rate_by_month,
)

st.set_page_config(page_title="Tendances", page_icon="📈", layout="wide")

df = load_movies()

st.title("📈 Tendances du cinéma (1980 — 2024)")
st.markdown(
    "Comment le cinéma a-t-il évolué sur 45 ans ? Quels genres dominent ? "
    "Y a-t-il une saisonnalité ? Les franchises ont-elles vraiment pris le pouvoir ? "
    "Explorez les tendances clés."
)
st.markdown("---")

# ===== Section 1 : timeline =====
st.header("⏳ Évolution dans le temps")

st.plotly_chart(films_timeline(df), use_container_width=True)
st.caption(
    "On observe une croissance régulière de la production de films jusqu'en 2019, "
    "suivie d'une chute en 2020 (effet COVID), puis une reprise."
)

st.plotly_chart(revenue_by_decade(df), use_container_width=True)
st.caption(
    "Les budgets et recettes médians explosent à partir des années 2000. "
    "L'inflation explique une partie, mais on voit aussi la concentration "
    "sur les blockbusters à gros budget."
)

st.markdown("---")

# ===== Section 2 : saisonnalité =====
st.header("📅 Saisonnalité — quand sortir un film ?")

st.plotly_chart(success_rate_by_month(df), use_container_width=True)
st.markdown(
    """
    **Insights** :
    - 🏖️ **Mai/Juin/Juillet** : la haute saison des blockbusters (Marvel, Pixar, suites de franchises)
    - 🎁 **Novembre/Décembre** : la saison des films familiaux et de prestige (Oscars)
    - 🍂 **Septembre/Octobre** : la zone "morte" du calendrier — beaucoup de films moyens
    """
)

st.markdown("---")

# ===== Section 3 : genres =====
st.header("🎭 Quels genres dominent ?")

c1, c2 = st.columns([1, 1])
with c1:
    st.plotly_chart(genre_distribution(df), use_container_width=True)
with c2:
    st.plotly_chart(genre_roi_comparison(df), use_container_width=True)

st.markdown(
    """
    **L'horreur, championne du ROI** : avec un budget moyen souvent inférieur à 10M$
    et des recettes qui peuvent dépasser 200M$ (cf. *Paranormal Activity*, *Blair Witch Project*),
    le genre offre les meilleurs retours sur investissement.

    **Animation et Aventure** : les budgets sont énormes, mais les succès quasi-garantis
    (très peu d'échecs dans ces genres).
    """
)

st.markdown("---")

# ===== Section 4 : franchises =====
st.header("🎬 Franchises vs Films originaux")

st.plotly_chart(franchise_vs_original(df), use_container_width=True)

franchise_rate = df[df["is_franchise"] == 1]["is_success"].mean() * 100
original_rate = df[df["is_franchise"] == 0]["is_success"].mean() * 100
franchise_rev = df[df["is_franchise"] == 1]["revenue"].mean()
original_rev = df[df["is_franchise"] == 0]["revenue"].mean()

c1, c2 = st.columns(2)
c1.metric("Taux de succès Franchise", f"{franchise_rate:.0f}%",
          f"+{franchise_rate - original_rate:.0f} pts vs original")
c2.metric("Revenue moyen Franchise", f"${franchise_rev/1e6:.0f}M",
          f"x{franchise_rev / original_rev:.1f} vs original")

st.markdown(
    """
    **Conclusion sans appel** : un film de franchise rapporte en moyenne **2.5x plus**
    qu'un film original et a **30 points de plus** de chances d'être rentable.
    Une explication évidente à la "franchisation" d'Hollywood.
    """
)

st.markdown("---")

# ===== Section 5 : top réalisateurs =====
st.header("👑 Top réalisateurs par recettes moyennes")

st.plotly_chart(director_top_chart(df, top_n=15), use_container_width=True)
st.caption(
    "Réalisateurs ayant au moins 3 films dans le dataset, triés par recettes moyennes. "
    "La couleur indique la note TMDB moyenne — les meilleurs réussissent à concilier qualité et succès."
)
