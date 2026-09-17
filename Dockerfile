# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README_PYPI.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel --outdir /dist

# Final minimal runtime image
FROM python:3.12-slim

LABEL org.opencontainers.image.title="SynthLine AI" \
      org.opencontainers.image.description="Synthetic visual-data generation for computer-vision teams." \
      org.opencontainers.image.source="https://github.com/thrivecodes/synthline-ai" \
      org.opencontainers.image.licenses="Apache-2.0"

# Install runtime system libraries for OpenCV and curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the built wheel and optional probe evaluation dependencies
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl "scikit-learn>=1.3" && \
    rm -rf /tmp/*.whl

# Create non-root user and persistent data volume directory
RUN useradd -m -u 1000 synthline && \
    mkdir -p /data && \
    chown -R synthline:synthline /data /app

USER synthline

ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /data

EXPOSE 8000

# Health check to monitor studio web server readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/projects || exit 1

ENTRYPOINT ["synthline-ai"]
CMD ["ui", "--host", "0.0.0.0", "--port", "8000"]
