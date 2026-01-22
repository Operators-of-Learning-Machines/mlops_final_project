import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "sclera_identity_classification"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)

@task(
    help={
        "experiment": "Hydra experiment to run (e.g. exp_1, exp_2)"
    }
)
def train(ctx: Context, experiment="exp_1") -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py experiment={experiment}", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run pytest tests/", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build "
        f"-f dockerfiles/train.dockerfile "
        f"-t sclera-train:gpu "
        f"--progress={progress} "
        f".",
        echo=True,
        pty=not WINDOWS,
    )
    # ctx.run(
    #     f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
    #     echo=True,
    #     pty=not WINDOWS
    # )

@task
def docker_build_frontend(ctx: Context, progress: str = "plain") -> None:
    """Build docker image for frontend."""
    ctx.run(
        f"docker build "
        f"-f dockerfiles/frontend.dockerfile "
        f"-t sclera-frontend:latest "
        f"--progress={progress} "
        f".",
        echo=True,
        pty=not WINDOWS,
    )

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
