FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_PYTHON=python3.12


WORKDIR /app

COPY pyproject.toml uv.lock ./

# IMPORTANT: no Alpine, no CPU fallback
RUN uv sync --frozen --no-install-project

COPY README.md README.md
COPY configs configs/
COPY models models/
COPY wandb wandb/
COPY tests tests/
COPY data data/


COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/sclera_identity_classification/train.py"]
