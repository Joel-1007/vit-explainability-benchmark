# Syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Prevent python from buffering stdout/stderr
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

# Create virtual environment and set it as default python
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pure dependencies first for optimal caching
COPY requirements.txt pyproject.toml README.md ./
RUN uv pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Run tests to ensure a valid environment
RUN uv pip install pytest numpy && pytest tests/

# Set the default command to the CLI runner
ENTRYPOINT ["python", "-m", "metrics.runner"]
CMD ["--help"]
