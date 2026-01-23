FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* ./

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi --no-root

COPY . .

# Create entrypoint that initializes DB and starts server
RUN cat > /app/entrypoint.sh << 'SCRIPT'
#!/bin/bash
set -e

# Initialize database (only if tables don't exist)
python -m src.db.init_db 2>/dev/null || echo "DB init skipped (tables may already exist)"

# Start the server
exec uvicorn src.main:app --host 0.0.0.0 --port 8000
SCRIPT

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

CMD ["/app/entrypoint.sh"]
