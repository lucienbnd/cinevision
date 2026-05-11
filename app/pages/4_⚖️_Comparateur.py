"""Page Comparateur — compare 2 films ou 2 réalisateurs côte à côte."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from components.charts import COLORS, get_top_directors, load_movies

st.set_page_config(page_title="Comparateur", page_icon="⚖️", layout="wide")

df = load_movies()

st.title("⚖️ Comparateur")
st.markdown(
    "Compare deux films ou deux réalisateurs côte à côte sur les indicateurs clés."
)
st.markdown("---")

mode = st.radio("Que veux-tu comparer ?", ["🎬 Deux films", "🎥 Deux réalisateurs"],
                horizontal=True)

# ==========================================================
# Mode 1 : Comparaison de deux films
# ==========================================================
if mode == "🎬 Deux films":
    df_sorted = df.sort_values("popularity", ascending=False)
    options = df_sorted.apply(
        lambda r: f"{r['title']} ({r['release_year']})", axis=1
    ).tolist()
    options_to_id = dict(zip(options, df_sorted["id"]))

    c1, c2 = st.columns(2)
    with c1:
        f1 = st.selectbox("Film 1", options, index=0, key="f1")
    with c2:
        f2 = st.selectbox("Film 2", options, index=1, key="f2")

    if f1 == f2:
        st.warning("⚠️ Choisis deux films différents.")
        st.stop()

    m1 = df[df["id"] == options_to_id[f1]].iloc[0]
    m2 = df[df["id"] == options_to_id[f2]].iloc[0]

    st.markdown("---")

    # Cartes d'identité côte à côte
    c1, c2 = st.columns(2)
    for col, m in zip([c1, c2], [m1, m2]):
        with col:
            st.markdown(f"### 🎬 {m['title']} ({m['release_year']})")
            if isinstance(m["tagline"], str) and m["tagline"]:
                st.caption(f"*« {m['tagline']} »*")
            st.write(f"**Réalisateur** : {m['director']}")
            st.write(f"**Genres** : {', '.join(m['genres'])}")
            st.write(f"**Durée** : {m['runtime']:.0f} minutes")
            st.write(f"**Casting** : {', '.join(m['cast_top5'][:3])}")
            st.metric("Budget", f"${m['budget']/1e6:.1f} M")
            st.metric("Recettes", f"${m['revenue']/1e6:.1f} M")
            st.metric("ROI", f"{m['roi']:+.2f}x")
            st.metric("Note TMDB", f"{m['vote_average']:.1f} / 10")

    st.markdown("---")

    # Radar chart de comparaison
    st.subheader("📊 Comparaison radar")

    def normalize(value, max_value):
        return min(value / max_value * 100, 100)

    categories = ["Budget", "Recettes", "ROI", "Note", "Popularité", "Durée"]

    def get_radar_values(m):
        return [
            normalize(m["budget"], 300e6),
            normalize(m["revenue"], 2e9),
            normalize(max(m["roi"], 0), 10),
            m["vote_average"] * 10,
            normalize(m["popularity"], 200),
            normalize(m["runtime"], 200),
        ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=get_radar_values(m1) + [get_radar_values(m1)[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=m1["title"],
        line_color=COLORS["primary"],
    ))
    fig.add_trace(go.Scatterpolar(
        r=get_radar_values(m2) + [get_radar_values(m2)[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=m2["title"],
        line_color=COLORS["secondary"],
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=550,
        title="Radar de comparaison (valeurs normalisées sur 100)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Mode 2 : Comparaison de deux réalisateurs
# ==========================================================
else:
    directors = get_top_directors(df, min_films=3)
    c1, c2 = st.columns(2)
    with c1:
        d1 = st.selectbox("Réalisateur 1", directors, index=0)
    with c2:
        d2 = st.selectbox(
            "Réalisateur 2", directors,
            index=1 if len(directors) > 1 else 0,
        )

    if d1 == d2:
        st.warning("⚠️ Choisis deux réalisateurs différents.")
        st.stop()

    df1 = df[df["director"] == d1]
    df2 = df[df["director"] == d2]

    st.markdown("---")

    c1, c2 = st.columns(2)
    for col, d, dfd in zip([c1, c2], [d1, d2], [df1, df2]):
        with col:
            st.markdown(f"### 🎥 {d}")
            st.metric("Nombre de films", len(dfd))
            st.metric("Budget total", f"${dfd['budget'].sum()/1e6:.1f} M")
            st.metric("Recettes totales", f"${dfd['revenue'].sum()/1e9:.2f} Mds")
            st.metric("Recettes moyennes / film", f"${dfd['revenue'].mean()/1e6:.1f} M")
            st.metric("ROI moyen", f"{dfd['roi'].mean():+.2f}x")
            st.metric("Note TMDB moyenne", f"{dfd['vote_average'].mean():.2f} / 10")
            st.metric("Taux de succès", f"{dfd['is_success'].mean()*100:.0f}%")

    st.markdown("---")

    # Filmographie
    st.subheader("🎬 Filmographies")
    c1, c2 = st.columns(2)
    cols_to_show = ["title", "release_year", "budget", "revenue", "roi", "vote_average"]
    with c1:
        st.markdown(f"**{d1}**")
        st.dataframe(
            df1[cols_to_show].sort_values("release_year"),
            use_container_width=True, hide_index=True,
        )
    with c2:
        st.markdown(f"**{d2}**")
        st.dataframe(
            df2[cols_to_show].sort_values("release_year"),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    # Radar chart
    st.subheader("📊 Comparaison radar")

    def normalize(value, max_value):
        return min(value / max_value * 100, 100)

    categories = ["Volume", "Recettes moy.", "ROI moy.", "Note moy.", "Taux succès"]

    def get_dir_values(dfd):
        return [
            normalize(len(dfd), 20),
            normalize(dfd["revenue"].mean(), 1e9),
            normalize(max(dfd["roi"].mean(), 0), 5),
            dfd["vote_average"].mean() * 10,
            dfd["is_success"].mean() * 100,
        ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=get_dir_values(df1) + [get_dir_values(df1)[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=d1,
        line_color=COLORS["primary"],
    ))
    fig.add_trace(go.Scatterpolar(
        r=get_dir_values(df2) + [get_dir_values(df2)[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=d2,
        line_color=COLORS["secondary"],
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)
