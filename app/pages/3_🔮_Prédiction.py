"""Page Prédiction — simulateur de succès cinématographique via ML."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from components.charts import gauge_chart, load_movies
from src.model import load_classifier, load_regressor, predict_movie

st.set_page_config(page_title="Prédiction", page_icon="🔮", layout="wide")

df = load_movies()

st.title("🔮 Simulateur de succès cinématographique")
st.markdown(
    "Configurez les caractéristiques de votre projet de film, et le modèle ML "
    "(Random Forest entraîné sur 2 976 films) estime ses chances de succès "
    "et ses recettes attendues."
)
st.markdown("---")

# ===== Chargement des modèles =====
@st.cache_resource
def load_models():
    return load_classifier(), load_regressor()

try:
    classifier, regressor = load_models()
except FileNotFoundError:
    st.error(
        "❌ Modèles ML non trouvés. Lancez d'abord `python src/model.py` "
        "pour entraîner et sauvegarder les modèles."
    )
    st.stop()

# ===== Formulaire =====
st.subheader("🎬 Configurez votre film")

col1, col2 = st.columns(2)

with col1:
    budget_m = st.slider("Budget (millions $)", 1, 500, 50)
    runtime = st.slider("Durée (minutes)", 60, 200, 110)
    release_year = st.slider("Année de sortie prévue", 2024, 2030, 2026)
    release_month = st.selectbox(
        "Mois de sortie",
        options=list(range(1, 13)),
        index=5,  # Juin par défaut (le meilleur mois)
        format_func=lambda m: ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
                               "Juillet", "Août", "Septembre", "Octobre",
                               "Novembre", "Décembre"][m - 1],
    )
    release_day_of_week = st.selectbox(
        "Jour de la semaine",
        options=list(range(7)),
        index=2,  # Mercredi par défaut (sortie classique en France)
        format_func=lambda d: ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi",
                               "Samedi", "Dimanche"][d],
    )

with col2:
    is_franchise = st.checkbox("Film de franchise / suite", value=False)
    is_international = st.checkbox("Co-production internationale", value=False)
    cast_popularity = st.slider("Popularité du casting (0 = inconnu, 100 = stars)", 0, 100, 20)
    director_experience = st.slider("Nb de films précédents du réalisateur", 0, 30, 3)
    director_avg_roi = st.slider(
        "ROI moyen des films précédents du réalisateur", -1.0, 10.0, 1.5, 0.1
    )
    director_avg_vote = st.slider(
        "Note moyenne des films précédents", 0.0, 10.0, 6.5, 0.1
    )

st.markdown("**Genre(s) du film** *(coche un ou plusieurs)*")
genre_cols = st.columns(6)
genre_options = ["Drama", "Action", "Comedy", "Thriller", "Adventure", "Crime",
                 "Science Fiction", "Fantasy", "Horror", "Family", "Romance", "Animation"]
selected_genres = []
for i, g in enumerate(genre_options):
    with genre_cols[i % 6]:
        if st.checkbox(g, value=(g == "Action"), key=f"g_{g}"):
            selected_genres.append(g)

st.markdown("---")

# ===== Bouton de prédiction =====
if st.button("🎯 Lancer la prédiction", type="primary", use_container_width=True):
    if not selected_genres:
        st.warning("⚠️ Sélectionne au moins un genre.")
        st.stop()

    # Construction du vecteur de features
    features = {
        "budget": budget_m * 1e6,
        "runtime": runtime,
        "release_year": release_year,
        "release_month": release_month,
        "release_day_of_week": release_day_of_week,
        "popularity": cast_popularity / 5,  # échelle TMDB approximative
        "vote_count": 0,
        "is_franchise": int(is_franchise),
        "nb_genres": len(selected_genres),
        "nb_production_companies": 3 if is_international else 1,
        "cast_avg_popularity": cast_popularity / 5,
        "lead_actor_popularity": cast_popularity / 4,
        "is_international": int(is_international),
        "director_nb_movies_prior": director_experience,
        "director_avg_vote_prior": director_avg_vote,
        "director_avg_roi_prior": director_avg_roi,
    }

    # One-hot genres
    for g in genre_options:
        col_name = f"genre_{g.lower().replace(' ', '_')}"
        features[col_name] = int(g in selected_genres)

    # Prédiction
    pred = predict_movie(features, classifier, regressor)

    st.markdown("---")
    st.subheader("📊 Résultats de la prédiction")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.plotly_chart(
            gauge_chart(pred["proba_success"] * 100, "Probabilité de succès"),
            use_container_width=True,
        )

    with c2:
        revenue_pred = pred["revenue_estimated"]
        roi_pred = (revenue_pred - budget_m * 1e6) / (budget_m * 1e6)
        st.markdown("### 💰 Estimation financière")
        st.metric("Recettes estimées", f"${revenue_pred/1e6:.1f} M")
        st.metric("Budget", f"${budget_m} M")
        st.metric("ROI estimé", f"{roi_pred:+.2f}x",
                  delta=f"{'Rentable' if roi_pred > 0 else 'Risque de perte'}")

    with c3:
        st.markdown("### 🎬 Synthèse")
        if pred["proba_success"] > 0.75:
            st.success(f"**🏆 Très fort potentiel**\n\nProbabilité de succès : {pred['proba_success']*100:.0f}%")
        elif pred["proba_success"] > 0.55:
            st.info(f"**👍 Potentiel correct**\n\nProbabilité de succès : {pred['proba_success']*100:.0f}%")
        elif pred["proba_success"] > 0.35:
            st.warning(f"**⚠️ Pari risqué**\n\nProbabilité de succès : {pred['proba_success']*100:.0f}%")
        else:
            st.error(f"**🚨 Très risqué**\n\nProbabilité de succès : {pred['proba_success']*100:.0f}%")

    # ===== Recommandations =====
    st.markdown("---")
    st.subheader("💡 Recommandations pour maximiser vos chances")

    recos = []
    if release_month in [9, 10, 1, 4]:
        recos.append("📅 **Mois de sortie** : septembre, octobre, janvier et avril ont historiquement les pires taux de succès. Envisage mai, juin, juillet ou décembre.")
    if not is_franchise:
        recos.append("🎬 **Franchise** : les films de franchise ont 30% de chances en plus d'être rentables. Penser à une saga ou une adaptation existante.")
    if cast_popularity < 30:
        recos.append("⭐ **Casting** : la popularité du casting est l'un des prédicteurs les plus forts. Investis dans des têtes d'affiche reconnues.")
    if budget_m > 200 and not is_franchise:
        recos.append("💰 **Gros budget sans franchise** : très risqué. Les blockbusters originaux échouent souvent. Privilégie une IP existante.")
    if "Horror" in selected_genres and budget_m > 30:
        recos.append("👻 **Horreur** : le secret du genre, c'est le petit budget. Réduis-le drastiquement (5-20 M$) pour maximiser le ROI.")
    if director_experience < 2:
        recos.append("🎥 **Réalisateur débutant** : risque accru. Pense à pairer avec un producteur expérimenté.")

    if not recos:
        st.success("✅ Ton projet coche déjà toutes les bonnes cases !")
    else:
        for r in recos:
            st.write(r)
else:
    st.info("👆 Configure les paramètres ci-dessus puis clique sur **Lancer la prédiction**.")
