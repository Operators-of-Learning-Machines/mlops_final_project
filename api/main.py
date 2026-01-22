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
import time
from prometheus_client import Counter, Histogram, make_asgi_app


CHANNELS = 3 # model input channels
MODEL_PATH = "models/model.pth" # statically set to the sclera model (note modify if multiple models are added)
MODEL_ONNX_PATH = "models/model.onnx"

root_counter = Counter("root_call", "Number of calls to the root endpoint")
successful_inference_counter = Counter("successful_inference", "Number of successfull calls to the inference endpoint")
failed_inference_counter = Counter("failed_inference", "Number of failed calls to the inference endpoint")
inference_requests_total = Counter(
    "sclera_inference_requests_total",
    "Total number of inference requests"
)

# Latency histogram (seconds)
inference_latency_seconds = Histogram(
    "sclera_inference_latency_seconds",
    "Time spent processing inference requests",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)


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
app.mount("/metrics", make_asgi_app())


@app.get("/")
def root():
    root_counter.inc()
    return {
        "message": "Welcome to the Sclera Identity Classification inference API!",
        "status": HTTPStatus.OK
    }



@app.post("/sclera_model")
async def sclera_model(file: UploadFile = File()):

    inference_requests_total.inc()
    start_time = time.perf_counter()
    
    try:

        img = await file.read()

        try:
            pil_img = Image.open(io.BytesIO(img))
            pil_img.load()  # force decoding now
        except UnidentifiedImageError:
            failed_inference_counter.inc()
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Please upload a valid PNG."
            )
        except DecompressionBombError:
            failed_inference_counter.inc()
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

        successful_inference_counter.inc()

        return {
            "result": output.flatten().tolist(),
            "status": HTTPStatus.OK
        }

    finally:
        elapsed = time.perf_counter() - start_time
        inference_latency_seconds.observe(elapsed)
        file.file.close()