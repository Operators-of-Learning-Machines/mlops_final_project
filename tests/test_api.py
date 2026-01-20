from fastapi.testclient import TestClient
from ..api.main import app
from http import HTTPStatus


client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.status == HTTPStatus.OK
    assert response.message == "Welcome to the Sclera Identity Classification inference API!"


def test_inference():
    test_img_path = "data/1_L/1L_l_1.png"
    response = client.post(
        "/sclera_model",
        files={"file": open(test_img_path)}
    )

    assert response.status_code == 200
    assert response.status == HTTPStatus.OK
