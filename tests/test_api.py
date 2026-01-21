from fastapi.testclient import TestClient
from api.main import app
from http import HTTPStatus
from PIL import Image
from sclera_identity_classification.data import ensure_data_present
from pathlib import Path
import pytest
import wandb


@pytest.fixture(scope="session")
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

client = TestClient(app)


@pytest.fixture
def define_img_limit():
    og_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 500
    yield
    Image.MAX_IMAGE_PIXELS = og_limit


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Sclera Identity Classification inference API!",
        "status": HTTPStatus.OK
    }

def test_file_upload():
    # Test for uploading a different file type than the expected png format:
    response = client.post(
        "sclera_model",
        files={
            "file": ("test.txt", b"Testing a different file format than png.", "text/plain")
        }
    )
    assert response.status_code == 400
    assert "Invalid image format. Please upload a valid PNG." == response.json()["detail"]

def test_large_file_upload(define_img_limit):
    ensure_data_present()
    # Test for the exception of uploading a way too large image:
    test_img_path = "data/1_L/1L_l_1.png"

    with open(test_img_path, "rb") as f:
        response = client.post(
            "/sclera_model",
            files={
                "file": ("1L_l_1.png", f, "image/png")
                }
        )

    assert response.status_code == 400
    assert "Image is too large." == response.json()["detail"]



def test_sclera_inference(pull_wandb):
    ensure_data_present() # Ensure data exists before defining test img path
    test_img_path = "data/1_L/1L_l_1.png"

    with open(test_img_path, "rb") as f:
        response = client.post(
            "/sclera_model",
            files={
                "file": ("1L_l_1.png", f, "image/png")
                }
        )

    res_json = response.json()
    assert response.status_code == 200
    assert res_json["status"] == HTTPStatus.OK
    assert res_json["result"]
