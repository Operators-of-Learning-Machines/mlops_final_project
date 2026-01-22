import pandas as pd
import requests
import streamlit as st
import torch

def classify_image(image_bytes, backend):
    """Send the image to the backend for inference."""
    url = f"{backend}/sclera_model"

    files = {
        "file": ("image.png", image_bytes, "image/png")
    }

    response = requests.post(url, files=files, timeout=10)

    if response.status_code != 200:
        st.error(f"Backend error {response.status_code}: {response.text}")
        return None

    return response.json()

def main() -> None:
    # temporary localhost backend for testing
    # backend = "http://localhost:8000"
    backend = "https://sclera-api-611901019822.europe-west1.run.app"

    st.title("Sclera Identity Classification")

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        return

    image_bytes = uploaded_file.read()
    result = classify_image(image_bytes, backend)

    if result is None:
        st.error("Inference failed")
        return

    logits = torch.tensor(result["result"], dtype=torch.float32)
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    probs = pd.Series(probs)


    st.image(image_bytes, caption="Uploaded Image")
    st.write("Top class:", probs.idxmax())

    df = pd.DataFrame(
        {
            "Class": [f"Class {i}" for i in range(len(probs))],
            "Probability": probs.values,
        }
    ).set_index("Class")

    st.bar_chart(df)


if __name__ == "__main__":
    main()
