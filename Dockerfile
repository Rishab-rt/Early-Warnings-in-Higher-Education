# Dockerfile - Early Warnings in Higher Education
#
# Builds and serves the team's existing Streamlit app (app.py) without
# modifying it. Trains the models at build time since the .pkl artifacts
# are gitignored (per the team's setup in model.py / README.md), so the
# container is self-contained and ready to run out of the box.

FROM python:3.12-slim

WORKDIR /app

# System deps needed by matplotlib/shap at build+runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (app.py, model.py, data_quality.py, data/, etc.)
COPY . .

# Train models at build time so logistic_model.pkl / decision_tree_model.pkl /
# scaler.pkl exist inside the image (they're .gitignore'd, not in the repo).
RUN python model.py

# Streamlit config
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
