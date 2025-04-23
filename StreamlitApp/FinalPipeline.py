import numpy as np
from PIL import Image
import streamlit as st
from Livrable1 import classify_image_external
from Livrable2 import denoise_image_external
from Livrable3 import generate_caption_external


def show_pipeline():

    st.title("Pipeline entière")

    st.divider()

    st.header("Étape 0 : Import de l'image")
    uploaded_file = st.file_uploader("Choisissez une image (format JPG ou PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0

        st.image(image, caption="Image originale importée", use_container_width=True)

        st.divider()

        st.header("Étape 1 : Classification (Photo vs Non-Photo)")
        is_photo = classify_image_external(image_np)

        if not is_photo:
            st.error("❌ Ce n’est pas une photo. Le pipeline s’arrête ici.")
            return
        else:
            st.success("✅ C’est bien une photo ! Étapes suivantes...")

        st.divider()

        st.header("Étape 2 : Prétraitement (bruit + débruitage)")
        apply_denoise = st.toggle("Appliquer le prétraitement", value=False)

        if apply_denoise:
            noise_amount = st.slider(
                "Intensité du bruit simulé (avant débruitage)",
                min_value=0.1, max_value=0.5, step=0.05, value=0.2
            )

            with st.spinner("Débruitage de l'image en cours..."):
                denoised_img = denoise_image_external(image_np, amount=noise_amount)

            st.image((denoised_img * 255).astype(np.uint8), caption="Image après prétraitement", use_container_width=True)
        else:
            denoised_img = image
            st.info("ℹ️ Prétraitement désactivé. L'image originale est utilisée pour le captioning.")

        st.divider()

        st.header("🖋️ Étape 3 : Génération automatique de légende")

        with st.spinner("Génération de légende..."):
            caption = generate_caption_external(denoised_img)

        st.success(f"Légende générée avec succès : {caption}")

        st.download_button("Télécharger la légende", data=caption, file_name="caption.txt", mime="text/plain")