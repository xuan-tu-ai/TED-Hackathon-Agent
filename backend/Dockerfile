# ── Backend Dockerfile ──────────────────────────────────────────
# Build stage: install dependencies with uv
# Run stage: lightweight Python runtime

# ================================================================
# BUILD
# ================================================================
FROM python:3.13-slim AS builder

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv (fast Python package installer)
RUN pip install --no-cache-dir uv

WORKDIR /app

# ── Install dependencies (layer caching) ───────────────────────
# Excludes 'anomaly-detection' group (torch, mlflow, etc.)
COPY pyproject.toml README.md ./
RUN uv sync --no-dev

# ── Copy application code ──────────────────────────────────────
# Excluded via .dockerignore: app/anomaly_detection, data_simulator, notebooks, tests
COPY agent_knowledge agent_knowledge/
COPY alembic.ini ./
COPY alembic/ alembic/
COPY app app/

# ================================================================
# RUNTIME
# ================================================================
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy .venv from builder (includes all runtime deps)
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/agent_knowledge /app/agent_knowledge
COPY --from=builder /app/alembic.ini /app/alembic.ini
COPY --from=builder /app/alembic /app/alembic
COPY --from=builder /app/app /app/app
COPY entrypoint.sh /app/entrypoint.sh

# Ensure the data directory for sqlite/chroma/sessions exists
RUN mkdir -p /app/data && chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
