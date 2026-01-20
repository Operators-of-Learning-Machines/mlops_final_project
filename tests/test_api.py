from fastapi.testclient import TestClient
from api.main import app
from http import HTTPStatus


client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Sclera Identity Classification inference API!",
        "status": HTTPStatus.OK
    }

def test_sclera_inference():
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
