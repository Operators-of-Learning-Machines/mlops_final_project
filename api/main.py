import torch
import io

from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from sclera_identity_classification.architectures.squeezenet import SqueezeNet
from PIL import Image
from http import HTTPStatus

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
async def sclera_model(data: UploadFile = File()):
    try:
        # Read uploaded image:
        img = await data.read()
        pil_img = Image.open(io.BytesIO(img))

        # Transform to correct format before forwarding to model:
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

        net = SqueezeNet(transfer_learning_model_path=MODEL_PATH, out_channels=220)
        net.to(device)

        with torch.inference_mode():
            output = net(input_img)
            # Format and send a response:
            response = {
                "result": output.flatten().tolist(),
                "status": HTTPStatus.OK,
            }
            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=e)
    finally:
        data.file.close() # To ensure the file is always closed
