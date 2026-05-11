# Image multi-arch (fonctionne sur ARM64 Oracle Free Tier ET x86 local)
FROM python:3.13-slim

# Bonnes pratiques Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dépendances système minimales (pour numpy/scipy/xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install des dépendances Python (cache-friendly)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copie du code et des données
COPY src/ ./src/
COPY app/ ./app/
COPY data/processed/movies_features.csv ./data/processed/movies_features.csv
COPY models/ ./models/

# Streamlit settings (désactive la télémétrie + browser tracking)
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Healthcheck pour Docker (Streamlit expose /_stcore/health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "app/🏠_Accueil.py"]
