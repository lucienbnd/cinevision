"""
Entraînement et évaluation des modèles ML pour CinéVision.

Ce module propose :
- Une tâche de classification : prédire si un film sera un succès (ROI > 1)
- Une tâche de régression : estimer les recettes (revenue)

Modèles comparés :
- Classification : Logistic Regression, Random Forest, XGBoost
- Régression : Linear Regression, Random Forest, XGBoost

Le meilleur modèle est sauvegardé dans models/.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Features de base utilisées pour le ML (hors one-hot genres)
NUMERIC_FEATURES = [
    "budget",
    "runtime",
    "release_year",
    "release_month",
    "release_day_of_week",
    "popularity",
    "vote_count",
    "is_franchise",
    "nb_genres",
    "nb_production_companies",
    "cast_avg_popularity",
    "lead_actor_popularity",
    "is_international",
    "director_nb_movies_prior",
    "director_avg_vote_prior",
    "director_avg_roi_prior",
]


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Construit la matrice de features (numériques + one-hot genres)."""
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    feature_cols = NUMERIC_FEATURES + genre_cols
    X = df[feature_cols].copy()
    # Gérer les NaN restants
    X = X.fillna(0)
    return X, feature_cols


def train_classification_models(df: pd.DataFrame) -> dict:
    """
    Entraîne et évalue 3 modèles de classification (prédire is_success).

    Returns:
        Dict avec les modèles entraînés et leurs métriques.
    """
    logger.info("=== Classification : prédire is_success (ROI > 1) ===")

    X, feature_cols = get_feature_matrix(df)
    y = df["is_success"]

    logger.info(f"Dataset : {X.shape[0]} films, {X.shape[1]} features")
    logger.info(f"Distribution cible : {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    results = {}
    for name, model in models.items():
        logger.info(f"\nEntraînement : {name}")

        # Logistic regression bénéficie du scaling, les autres non
        if name == "logistic_regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Métriques
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Validation croisée 5-fold sur F1
        X_for_cv = X_train_scaled if name == "logistic_regression" else X_train
        cv_scores = cross_val_score(model, X_for_cv, y_train, cv=5, scoring="f1", n_jobs=-1)

        logger.info(f"  Accuracy : {acc:.3f} | Precision : {prec:.3f} | "
                    f"Recall : {rec:.3f} | F1 : {f1:.3f}")
        logger.info(f"  F1 cross-val (5-fold) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "confusion_matrix": cm,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    # Sélection du meilleur modèle
    best_name = max(results, key=lambda k: results[k]["f1"])
    logger.info(f"\n>>> Meilleur modèle classification : {best_name} "
                f"(F1 = {results[best_name]['f1']:.3f})")

    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": results[best_name]["model"],
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


def train_regression_models(df: pd.DataFrame) -> dict:
    """
    Entraîne et évalue 3 modèles de régression (prédire revenue).

    On utilise log(revenue) pour stabiliser les variances (le revenue varie sur
    plusieurs ordres de grandeur).
    """
    logger.info("=== Régression : prédire log(revenue) ===")

    X, feature_cols = get_feature_matrix(df)
    y = np.log1p(df["revenue"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        logger.info(f"\nEntraînement : {name}")

        if name == "linear_regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Métriques sur log(revenue)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Métriques sur revenue réel (après expm1)
        y_test_real = np.expm1(y_test)
        y_pred_real = np.expm1(y_pred)
        mae_real = mean_absolute_error(y_test_real, y_pred_real)

        logger.info(f"  RMSE (log) : {rmse:.3f} | MAE (log) : {mae:.3f} | R² : {r2:.3f}")
        logger.info(f"  MAE (revenue réel) : ${mae_real:,.0f}")

        results[name] = {
            "model": model,
            "rmse_log": rmse,
            "mae_log": mae,
            "r2": r2,
            "mae_real": mae_real,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    best_name = max(results, key=lambda k: results[k]["r2"])
    logger.info(f"\n>>> Meilleur modèle régression : {best_name} "
                f"(R² = {results[best_name]['r2']:.3f})")

    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": results[best_name]["model"],
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


def save_best_models(clf_results: dict, reg_results: dict) -> None:
    """Sauvegarde les meilleurs modèles classification et régression."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    clf_path = MODELS_DIR / "best_classifier.pkl"
    joblib.dump(
        {
            "model": clf_results["best_model"],
            "scaler": clf_results["scaler"],
            "feature_cols": clf_results["feature_cols"],
            "model_name": clf_results["best_model_name"],
        },
        clf_path,
    )
    logger.info(f"Modèle classification sauvegardé : {clf_path}")

    reg_path = MODELS_DIR / "best_regressor.pkl"
    joblib.dump(
        {
            "model": reg_results["best_model"],
            "scaler": reg_results["scaler"],
            "feature_cols": reg_results["feature_cols"],
            "model_name": reg_results["best_model_name"],
        },
        reg_path,
    )
    logger.info(f"Modèle régression sauvegardé : {reg_path}")


def load_classifier(path: Path = None) -> dict:
    """Charge le modèle de classification sauvegardé."""
    if path is None:
        path = MODELS_DIR / "best_classifier.pkl"
    return joblib.load(path)


def load_regressor(path: Path = None) -> dict:
    """Charge le modèle de régression sauvegardé."""
    if path is None:
        path = MODELS_DIR / "best_regressor.pkl"
    return joblib.load(path)


def predict_movie(features: dict, classifier: dict, regressor: dict) -> dict:
    """
    Prédit le succès et les recettes d'un film hypothétique.

    Args:
        features: dict des features du film (doit contenir les colonnes attendues)
        classifier: dict chargé via load_classifier()
        regressor: dict chargé via load_regressor()

    Returns:
        Dict avec proba_success, revenue_estimated, is_success_predicted.
    """
    feature_cols = classifier["feature_cols"]
    X = pd.DataFrame([features])[feature_cols].fillna(0)

    # Classification
    clf = classifier["model"]
    if classifier["model_name"] == "logistic_regression":
        X_scaled = classifier["scaler"].transform(X)
        proba = clf.predict_proba(X_scaled)[0, 1]
        is_success = int(clf.predict(X_scaled)[0])
    else:
        proba = clf.predict_proba(X)[0, 1]
        is_success = int(clf.predict(X)[0])

    # Régression
    reg = regressor["model"]
    if regressor["model_name"] == "linear_regression":
        X_scaled = regressor["scaler"].transform(X)
        log_rev = reg.predict(X_scaled)[0]
    else:
        log_rev = reg.predict(X)[0]
    revenue = np.expm1(log_rev)

    return {
        "proba_success": float(proba),
        "is_success_predicted": is_success,
        "revenue_estimated": float(revenue),
    }


def get_feature_importance(model_dict: dict, top_n: int = 15) -> pd.DataFrame:
    """Retourne l'importance des features pour un modèle entraîné."""
    model = model_dict["model"]
    feature_cols = model_dict["feature_cols"]

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).flatten()
    else:
        return pd.DataFrame()

    df_imp = pd.DataFrame({"feature": feature_cols, "importance": importance})
    return df_imp.sort_values("importance", ascending=False).head(top_n)


def main_pipeline(input_file: str = "movies_features.csv") -> None:
    """Pipeline complet : chargement + entraînement + sauvegarde."""
    filepath = DATA_PROCESSED_DIR / input_file
    logger.info(f"Chargement de {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset : {df.shape[0]} films, {df.shape[1]} colonnes")

    clf_results = train_classification_models(df)
    reg_results = train_regression_models(df)

    save_best_models(clf_results, reg_results)

    logger.info("\n=== PIPELINE ML TERMINÉ ===")
    logger.info(f"Best classifier : {clf_results['best_model_name']} "
                f"(F1 = {clf_results['results'][clf_results['best_model_name']]['f1']:.3f})")
    logger.info(f"Best regressor : {reg_results['best_model_name']} "
                f"(R² = {reg_results['results'][reg_results['best_model_name']]['r2']:.3f})")


if __name__ == "__main__":
    main_pipeline()
