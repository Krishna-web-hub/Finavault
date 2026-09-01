# Multi-stage build. The heavy part of this image is not FinVault's own code
# — it is torch and sentence-transformers, pulled in by the local embedding
# provider (ingestion/embeddings.py), which is what keeps document text from
# ever reaching a third-party embedding API. Building dependencies in a
# separate stage means an application code change rebuilds only the last few
# layers instead of recompiling that whole tree.

# ---------- builder ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential is needed to compile psycopg2 from source; it stays in this
# stage and never reaches the runtime image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Dependency metadata first, sources second. Docker caches by layer, so this
# ordering means editing a .py file does not invalidate the (slow) dependency
# install above it.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install .

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Where sentence-transformers caches model weights. Pointed at a writable
    # path under the app user's home rather than the default root-owned
    # location, which a non-root container cannot write to.
    HF_HOME=/home/finvault/.cache/huggingface \
    PORT=7860

# libpq5 is psycopg2's runtime dependency — the client library, not the
# compiler toolchain that built against it.
# Non-root user with UID 1000 (Hugging Face Spaces default non-root UID)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 finvault

COPY --from=builder /opt/venv /opt/venv
COPY --chown=finvault:finvault frontend/ /app/frontend/
COPY --chown=finvault:finvault src/ /app/src/

WORKDIR /app
RUN mkdir -p /app/.secrets /app/data /home/finvault/.cache \
 && chown -R finvault:finvault /app /home/finvault

USER finvault

EXPOSE 7860

# A liveness probe the orchestrator can use before it has an ingress.
HEALTHCHECK --interval=30s --timeout=3s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# Start Uvicorn on port 7860 (Hugging Face Spaces standard) or dynamically via $PORT
CMD ["sh", "-c", "uvicorn finvault.api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]

