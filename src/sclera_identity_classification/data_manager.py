# data_manager.py
import json
import hashlib
from datetime import datetime, timezone
from google.cloud import storage
import uuid

BUCKET_NAME = "sclera-api-logging"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def log_sclera_request(
    *,
    image_bytes: bytes,
    filename: str,
    predicted_class: int,
    confidence: float,
    model_version: str = "sclera_v1"
):
    ts = datetime.now(timezone.utc)

    record = {
        "timestamp": ts.isoformat(),
        "filename": filename,
        "input": {
            "sha256": sha256_bytes(image_bytes),
            "image_size": [224, 224],
        },
        "output": {
            "class": predicted_class,
            "confidence": confidence,
        },
        "model": {
            "format": "onnx",
            "version": model_version,
        },
    }

    date_prefix = ts.strftime("%Y-%m-%d")
    object_name = (
        f"sclera_model/{date_prefix}/"
        f"{ts.strftime('%H-%M-%S')}_{uuid.uuid4().hex}.json"
    )

    blob = bucket.blob(object_name)
    blob.upload_from_string(
        json.dumps(record),
        content_type="application/json"
    )
