# ForecastIQ - single container (API + web app), built for Hugging Face Spaces (Docker SDK)
# but runs anywhere Docker runs:  docker build -t forecastiq . && docker run -p 7860:7860 forecastiq
FROM python:3.11-slim

# xgboost needs the OpenMP runtime, which the slim image does not include
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_ARENA_MAX=2
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user backend ./backend
COPY --chown=user src ./src
COPY --chown=user web ./web
COPY --chown=user artifacts ./artifacts

# Hugging Face uses port 7860; Render (and most hosts) inject $PORT - this honours both.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips '*'"]
