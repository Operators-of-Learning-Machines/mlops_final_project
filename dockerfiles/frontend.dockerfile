FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim


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

RUN uv sync --frozen --no-install-project

COPY src src/
COPY README.md README.md
COPY frontend frontend/


RUN uv sync --frozen

ENTRYPOINT [ "uv", "run", "streamlit", "run", "frontend/main.py", "--server.port=${PORT}" ]
