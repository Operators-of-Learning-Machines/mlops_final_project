import wandb
from pathlib import Path


def pull_wandb():
    """
    Pulls the latest trained model artifact from Weights & Biases
    and returns the local path to the .pth file.
    """

    api = wandb.Api()

    # Replace with your actual entity/project if needed
    entity = None  # or "your_wandb_entity"
    project = "sclera-identity-classification"

    artifact_name = "sclera-identity-classification-model:latest"

    artifact = api.artifact(f"{entity}/{project}/{artifact_name}" if entity else f"{project}/{artifact_name}")

    artifact_dir = artifact.download(root='models')

    # Find the .pth file
    model_files = list(Path(artifact_dir).glob("*.pth"))
    if not model_files:
        raise FileNotFoundError("No .pth model file found in W&B artifact")

    return model_files[0]
