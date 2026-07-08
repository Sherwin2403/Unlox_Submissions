
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MODEL_PATH = "cifar10_cnn.keras"

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((32, 32))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 32, 32, 3)

st.title("🖼️ CIFAR-10 Image Classifier")
st.write(
    "Upload an image and the CNN will predict which of the 10 CIFAR-10 "
    "classes it belongs to: " + ", ".join(CLASS_NAMES)
)

try:
    model = load_model()
except Exception as e:
    st.error(
        f"Couldn't load `{MODEL_PATH}`. Make sure you've run `cifar10_cnn.py` "
        f"first so the model file exists.\n\nError: {e}"
    )
    st.stop()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    input_arr = preprocess(image)
    preds = model.predict(input_arr, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:3]

    with col2:
        st.subheader("Predictions")
        for i in top_idx:
            st.write(f"**{CLASS_NAMES[i]}**")
            st.progress(float(preds[i]))
            st.caption(f"{preds[i] * 100:.2f}% confidence")

    st.info(
        "Note: CIFAR-10 images are only 32×32px, so the model was trained on "
        "very low-res data — accuracy on high-res photos of unrelated subjects "
        "may be lower than the ~85-88% test accuracy reported during training."
    )
else:
    st.caption("Waiting for an image upload...")
