FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    gcc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/sclera_identity_classification
