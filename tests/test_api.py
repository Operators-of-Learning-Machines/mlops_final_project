from fastapi.testclient import TestClient
from api.main import app
from http import HTTPStatus
from PIL import Image
from sclera_identity_classification.data import ensure_data_present
import pytest
from models.ensure_model_pulled import pull_wandb 


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def define_img_limit():
    og_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 500
    yield
    Image.MAX_IMAGE_PIXELS = og_limit


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Sclera Identity Classification inference API!",
        "status": HTTPStatus.OK
    }

def test_file_upload(client):
    # Test for uploading a different file type than the expected png format:
    response = client.post(
        "sclera_model",
        files={
            "file": ("test.txt", b"Testing a different file format than png.", "text/plain")
        }
    )
    assert response.status_code == 400
    assert "Invalid image format. Please upload a valid PNG." == response.json()["detail"]

def test_large_file_upload(define_img_limit, client):
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



def test_sclera_inference(client):
    ensure_data_present() # Ensure data exists before defining test img path
    pull_wandb()
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
