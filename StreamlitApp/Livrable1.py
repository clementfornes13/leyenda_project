import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

def show_livrable1():
    st.title("Classification d'Images - Photo vs Non-Photo")

    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("model/binary_classifier.h5")

    model = load_model()

    def classify_image(image_array):
        resized = tf.image.resize(image_array, (256, 256)) / 255.0
        input_tensor = np.expand_dims(resized, axis=0)
        return model.predict(input_tensor)[0][0]

    uploaded_files = st.file_uploader("Importez des images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(image).astype(np.float32)

            st.image(image, caption="Image importée", use_container_width=True)

            score = classify_image(image_array)
            label = "📸 Photo" if score > 0.5 else "Non-Photo"
            confidence = score if score > 0.5 else 1 - score

            st.markdown(f"**Classe prédite :** {label}")
            st.markdown(f"**Confiance :** {confidence:.2%}")
            st.markdown("---")

def classify_image_external(image_array):
    model = tf.keras.models.load_model("model/binary_classifier.h5")
    resized = tf.image.resize(image_array, (256, 256)) / 255.0
    input_tensor = np.expand_dims(resized, axis=0)
    return model.predict(input_tensor)[0][0]