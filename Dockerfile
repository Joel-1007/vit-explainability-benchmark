# syntax=docker/dockerfile:1
# Python 3.13 + uv. PyPI torch wheels for Linux bundle the CUDA runtime, so the
# host only needs an NVIDIA driver (run with `docker run --gpus all ...`).
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Make `uv sync` / `uv run` (used by run_pilot.sh, run_benchmark.sh) use this env
    UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install locked dependencies first (layer caching)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# Copy the project and install it into the same env
COPY . .
RUN uv sync --frozen

# Default command: run the pilot (smoke test)
CMD ["bash", "run_pilot.sh"]
