# Build stage
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache ".[all]"

COPY src/ src/

# Test stage — run in CI to gate the build
FROM builder AS test
RUN uv pip install --system --no-cache ".[dev]"
COPY tests/ tests/
RUN python -m pytest tests/unit/ -q --no-header

# Runtime stage — distroless for minimal attack surface
FROM gcr.io/distroless/python3-debian12 AS runtime

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /build/src /app/src

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Library — no entrypoint. Consumers import it directly.
CMD ["python", "-c", "import actrone_memory; print(actrone_memory.__version__)"]
