import torch
import io

from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from sclera_identity_classification.architectures.squeezenet import SqueezeNet
from http import HTTPStatus
import os
from models.ensure_model_pulled import pull_wandb
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError


CHANNELS = 3 # model input channels
MODEL_PATH = "models/model.pth" # statically set to the sclera model (note modify if multiple models are added)

app = FastAPI()

@app.get("/")
def root():
    return {
        "message": "Welcome to the Sclera Identity Classification inference API!",
        "status": HTTPStatus.OK
    }


@app.post("/sclera_model")
async def sclera_model(file: UploadFile = File()):
    try:
        if not os.path.exists(MODEL_PATH):
            pull_wandb()

        img = await file.read()

        try:
            pil_img = Image.open(io.BytesIO(img))
            pil_img.load()  # force decoding now
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Please upload a valid PNG."
            )
        except DecompressionBombError:
            raise HTTPException(
                status_code=400,
                detail="Image is too large."
            )

        base_transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_img = base_transform(pil_img).to(device)
        input_img = input_img.unsqueeze(0)

        net = SqueezeNet(
            transfer_learning_model_path=MODEL_PATH,
            out_channels=220
        ).to(device)

        with torch.inference_mode():
            output = net(input_img)

        return {
            "result": output.flatten().tolist(),
            "status": HTTPStatus.OK
        }

    finally:
        file.file.close()