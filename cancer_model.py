import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin_cancer_model.keras")

# DermaMNIST class labels (index -> name)
CLASS_NAMES = [
    "Actinic keratoses / intraepithelial carcinoma",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((28, 28))
    arr = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)  # shape (1, 28, 28, 3)


st.title("🩺 Skin Lesion Classifier")
st.caption("Educational demo trained on DermaMNIST (28x28 images, 7 classes).")
st.warning(
    "This tool is NOT a medical diagnosis. Always consult a qualified "
    "dermatologist about any skin concern."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file `skin_cancer_model.keras` was not found. Make sure it is "
        "in the same folder as this script in your GitHub repo."
    )
    st.stop()

model = load_model()

uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    probs = model.predict(preprocess(image), verbose=0)[0]
    top = int(np.argmax(probs))

    st.subheader("Prediction")
    st.success(f"{CLASS_NAMES[top]} ({probs[top] * 100:.1f}% confidence)")

    chart_df = pd.DataFrame({"Probability": probs}, index=CLASS_NAMES)
    st.bar_chart(chart_df)
