# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv for deterministic dependency management
RUN pip install uv

# Create virtual environment and activate it
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy dependency files first (layer caching)
COPY requirements.txt pyproject.toml uv.lock ./

# Install all Python dependencies
RUN uv pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command: run the pilot (smoke test)
CMD ["bash", "run_pilot.sh"]
