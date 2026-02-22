# ──────────────────────────────────────────────
# FinScope AI - Multi-stage Production Dockerfile
# ──────────────────────────────────────────────

# ── Stage 1: Builder ──
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec
COPY pyproject.toml .

# Install Python packages to a prefix
RUN pip install --no-cache-dir --prefix=/install .


# ── Stage 2: Runtime ──
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create directories
RUN mkdir -p logs models/artifacts data/raw data/processed

# Create non-root user
RUN groupadd -r finscope && useradd -r -g finscope finscope \
    && chown -R finscope:finscope /app
USER finscope

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
