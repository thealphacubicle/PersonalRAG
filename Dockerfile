# Dockerfile
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Use HTTPS mirrors and retry to avoid transient 5xx errors
RUN set -eux; \
    sed -i -E 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends ca-certificates libgomp1 libstdc++6; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY src/ ./src/

EXPOSE 8000
CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
