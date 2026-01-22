import time
from locust import HttpUser, task, between


IMG_PATH = "data/1_L/1L_l_1.png" # Locally saved test image...

with open(IMG_PATH, "rb") as f:
    IMG_DATA = f.read()

class TestUser(HttpUser):

    wait_time = between(1, 2) # waits 1-10 seconds between each request

    @task
    def get_root(self):
        self.client.get("/")


    @task
    def post_inference(self):
        self.client.post(
            "/sclera_model",
            files={
                    "file": ("1L_l_1.png", IMG_DATA, "image/png")
                }
        )
