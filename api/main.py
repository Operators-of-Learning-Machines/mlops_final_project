import io

from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from http import HTTPStatus
import os
from models.ensure_model_pulled import pull_wandb
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError
from models import onnx as onnx_module
from contextlib import asynccontextmanager



CHANNELS = 3 # model input channels
MODEL_PATH = "models/model.pth" # statically set to the sclera model (note modify if multiple models are added)
MODEL_ONNX_PATH = "models/model.onnx"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at application startup.
    Ensures the ONNX model exists and loads it into memory.
    """

    if not os.path.exists(MODEL_ONNX_PATH):
        if not os.path.exists(MODEL_PATH):
            pull_wandb()

        onnx_module.export_to_onnx(
            pth_path=MODEL_PATH,
            onnx_path=MODEL_ONNX_PATH
        )

    onnx_module.load_onnx_session()

    yield



app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {
        "message": "Welcome to the Sclera Identity Classification inference API!",
        "status": HTTPStatus.OK
    }


@app.post("/sclera_model")
async def sclera_model(file: UploadFile = File()):
    try:

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
                transforms.Resize((224, 224)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )


        input_img = base_transform(pil_img)
        input_img = input_img.unsqueeze(0)

        # ONNX expects numpy arrays
        input_np = input_img.numpy()

        # Run inference
        output = onnx_module.onnx_session.run(
            None,
            {"input": input_np}
        )[0]

        return {
            "result": output.flatten().tolist(),
            "status": HTTPStatus.OK
        }

    finally:
        file.file.close()