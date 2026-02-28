# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some packages (torch, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency manifest first for layer caching
COPY pyproject.toml ./

# Install all dependencies into an isolated prefix
RUN pip install --upgrade pip && \
    pip install --prefix=/install . && \
    pip install --prefix=/install pysqlite3-binary

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps (git is needed by agents for cloning repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./

# Create runtime directories (mounted as volumes in compose)
RUN mkdir -p cloned_repo downloads logs checkpoints

# Default environment
ENV PYTHONUNBUFFERED=1
ENV STORAGE_BACKEND=local

CMD ["python", "-m", "src.cli"]
