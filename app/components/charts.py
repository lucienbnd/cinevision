"""
Composants graphiques réutilisables pour l'application Streamlit.

Toutes les visualisations utilisent Plotly pour bénéficier de l'interactivité
(zoom, tooltips, légendes cliquables).
"""

import ast
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "movies_features.csv"

# Palette CinéVision
COLORS = {
    "primary": "#e63946",
    "secondary": "#1d3557",
    "accent": "#f4a261",
    "success": "#06a77d",
    "danger": "#ff006e",
    "neutral": "#264653",
}


@st.cache_data
def load_movies() -> pd.DataFrame:
    """Charge le dataset des films (mis en cache par Streamlit)."""
    df = pd.read_csv(DATA_FILE)
    df["release_date"] = pd.to_datetime(df["release_date"])
    for col in ["genres", "production_companies", "cast_top5", "cast_top5_popularity"]:
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
        )
    return df


def get_all_genres(df: pd.DataFrame) -> list[str]:
    """Retourne la liste triée de tous les genres présents."""
    all_g = sorted({g for genres in df["genres"] for g in genres})
    return all_g


def get_top_directors(df: pd.DataFrame, min_films: int = 3) -> list[str]:
    """Retourne les réalisateurs avec au moins min_films films."""
    counts = df.groupby("director").size()
    return sorted(counts[counts >= min_films].index.tolist())


def budget_revenue_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot interactif budget vs recettes."""
    fig = px.scatter(
        df,
        x="budget",
        y="revenue",
        color="vote_average",
        size="popularity",
        hover_name="title",
        hover_data={"release_year": True, "main_genre": True,
                    "director": True, "budget": ":,.0f", "revenue": ":,.0f"},
        log_x=True,
        log_y=True,
        color_continuous_scale="Viridis",
        labels={"budget": "Budget ($)", "revenue": "Recettes ($)",
                "vote_average": "Note"},
        title="Budget vs Recettes (taille = popularité, couleur = note)",
    )
    fig.add_shape(
        type="line",
        x0=1e3, y0=1e3, x1=1e9, y1=1e9,
        line=dict(color="red", dash="dash", width=1),
    )
    fig.update_layout(height=550)
    return fig


def revenue_by_decade(df: pd.DataFrame) -> go.Figure:
    """Évolution des recettes médianes par décennie."""
    by_dec = df.groupby("release_decade").agg(
        revenue_median=("revenue", "median"),
        budget_median=("budget", "median"),
        n_films=("id", "count"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_dec["release_decade"],
        y=by_dec["budget_median"] / 1e6,
        name="Budget médian",
        marker_color=COLORS["secondary"],
    ))
    fig.add_trace(go.Bar(
        x=by_dec["release_decade"],
        y=by_dec["revenue_median"] / 1e6,
        name="Recettes médianes",
        marker_color=COLORS["primary"],
    ))
    fig.update_layout(
        title="Évolution des budgets et recettes médians par décennie",
        xaxis_title="Décennie",
        yaxis_title="Millions $",
        barmode="group",
        height=450,
    )
    return fig


def success_rate_by_month(df: pd.DataFrame) -> go.Figure:
    """Taux de succès par mois de sortie."""
    months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
              "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    stats = df.groupby("release_month").agg(
        success_rate=("is_success", "mean"),
        revenue_mean=("revenue", "mean"),
        n=("id", "count"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[months[i - 1] for i in stats["release_month"]],
        y=stats["success_rate"] * 100,
        marker_color=COLORS["primary"],
        text=[f"{r*100:.0f}%" for r in stats["success_rate"]],
        textposition="outside",
        name="Taux de succès",
    ))
    fig.add_hline(
        y=df["is_success"].mean() * 100,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Moyenne : {df['is_success'].mean()*100:.0f}%",
    )
    fig.update_layout(
        title="Taux de succès des films selon le mois de sortie",
        xaxis_title="Mois de sortie",
        yaxis_title="Taux de succès (%)",
        height=450,
    )
    return fig


def genre_distribution(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Distribution des films par genre (treemap)."""
    df_g = df.explode("genres").rename(columns={"genres": "genre"})
    counts = df_g["genre"].value_counts().head(top_n).reset_index()
    counts.columns = ["genre", "count"]

    fig = px.treemap(
        counts,
        path=["genre"],
        values="count",
        color="count",
        color_continuous_scale="Viridis",
        title=f"Répartition des films par genre (top {top_n})",
    )
    fig.update_layout(height=450)
    return fig


def genre_roi_comparison(df: pd.DataFrame) -> go.Figure:
    """ROI médian par genre."""
    df_g = df.explode("genres").rename(columns={"genres": "genre"})
    stats = df_g.groupby("genre").agg(
        roi_median=("roi", "median"),
        success_rate=("is_success", "mean"),
        n=("id", "count"),
    ).reset_index()
    stats = stats[stats["n"] >= 50].sort_values("roi_median", ascending=True)

    fig = px.bar(
        stats,
        x="roi_median",
        y="genre",
        orientation="h",
        color="success_rate",
        color_continuous_scale="RdYlGn",
        labels={"roi_median": "ROI médian", "genre": "Genre",
                "success_rate": "Taux succès"},
        title="ROI médian par genre (≥ 50 films)",
        text="roi_median",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=550)
    return fig


def films_timeline(df: pd.DataFrame) -> go.Figure:
    """Nombre de films par année."""
    yearly = df.groupby("release_year").size().reset_index(name="count")
    fig = px.bar(
        yearly,
        x="release_year",
        y="count",
        title="Nombre de films par année (dans notre échantillon)",
        labels={"release_year": "Année", "count": "Nombre de films"},
        color="count",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
    return fig


def franchise_vs_original(df: pd.DataFrame) -> go.Figure:
    """Comparaison franchise vs original (3 KPIs)."""
    stats = df.groupby("is_franchise").agg(
        budget_mean=("budget", "mean"),
        revenue_mean=("revenue", "mean"),
        success_rate=("is_success", "mean"),
    ).reset_index()
    stats["type"] = stats["is_franchise"].map({0: "Original", 1: "Franchise"})

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stats["type"], y=stats["budget_mean"] / 1e6,
        name="Budget moyen (M$)",
        marker_color=COLORS["secondary"],
    ))
    fig.add_trace(go.Bar(
        x=stats["type"], y=stats["revenue_mean"] / 1e6,
        name="Revenue moyen (M$)",
        marker_color=COLORS["primary"],
    ))
    fig.update_layout(
        title="Franchise vs Film original — budgets et recettes",
        yaxis_title="Millions $",
        barmode="group",
        height=400,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap de corrélation des variables clés."""
    cols = ["budget", "revenue", "roi", "runtime", "vote_average", "vote_count",
            "popularity", "cast_avg_popularity", "is_franchise",
            "director_avg_roi_prior"]
    corr = df[cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Matrice de corrélation",
        aspect="auto",
    )
    fig.update_layout(height=550)
    return fig


def director_top_chart(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Top réalisateurs par recettes moyennes."""
    stats = df.groupby("director").agg(
        n_films=("id", "count"),
        revenue_mean=("revenue", "mean"),
        roi_mean=("roi", "mean"),
        vote_avg=("vote_average", "mean"),
    )
    stats = stats[stats["n_films"] >= 3].sort_values("revenue_mean", ascending=False).head(top_n)

    fig = px.bar(
        stats.reset_index(),
        x="revenue_mean",
        y="director",
        orientation="h",
        color="vote_avg",
        color_continuous_scale="Viridis",
        labels={"revenue_mean": "Recettes moyennes ($)", "director": "Réalisateur",
                "vote_avg": "Note moyenne"},
        title=f"Top {top_n} réalisateurs par recettes moyennes (≥ 3 films)",
        text="n_films",
    )
    fig.update_traces(texttemplate="%{text} films", textposition="outside")
    fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
    return fig


def gauge_chart(value: float, title: str, max_value: float = 100) -> go.Figure:
    """Jauge de prédiction (probabilité de succès)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [0, max_value]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [0, 33], "color": "#ff6b6b"},
                {"range": [33, 66], "color": "#feca57"},
                {"range": [66, 100], "color": "#06a77d"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
        number={"suffix": "%" if max_value == 100 else ""},
    ))
    fig.update_layout(height=300)
    return fig
