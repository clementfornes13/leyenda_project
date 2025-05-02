import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import io

def show_livrable2():
    st.title("Débruitage d'image avec U-Net")

    @st.cache_resource
    def load_model():
        model = tf.keras.models.load_model("../StreamlitApp/model/salt_noise.h5", compile=False)
        model.compile(optimizer="adam", loss="mse")
        return model

    model = load_model()

    def add_salt_noise(image_np, amount=0.05):
        output = np.copy(image_np)
        total_pixels = image_np.shape[0] * image_np.shape[1]
        num_salt = int(amount * total_pixels)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image_np.shape[:2]]
        output[coords[0], coords[1], :] = 1
        return np.clip(output, 0, 1)

    def denoise_image(noisy_img):
        input_img = np.expand_dims(noisy_img, axis=0)
        denoised = model.predict(input_img)[0]
        return np.clip(denoised, 0., 1.)

    def display_results(original, noisy, denoised, key):
        col1, col2, col3 = st.columns(3)
        col1.image((original * 255).astype(np.uint8), caption="Image originale", use_container_width=True)
        col2.image((noisy * 255).astype(np.uint8), caption="Image bruitée", use_container_width=True)
        col3.image((denoised * 255).astype(np.uint8), caption="Image débruitée", use_container_width=True)

        image_psnr = psnr(original, denoised, data_range=1.0)
        image_ssim = np.mean([
            ssim(original[:, :, c], denoised[:, :, c], data_range=1.0)
            for c in range(3)
        ])
        st.markdown(f"**PSNR :** {image_psnr:.2f} dB")
        st.markdown(f"**SSIM :** {image_ssim:.4f}")

        denoised_pil = Image.fromarray((denoised * 255).astype(np.uint8))
        buf = io.BytesIO()
        denoised_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button("Télécharger l'image débruitée", data=byte_im, file_name=f"denoised_{key}.png", mime="image/png", key=f"dl_{key}")

    # Création des onglets principaux
    page1, page2 = st.tabs([
        "Modèle",
        "Performance générales du modèle",
    ])

    # Onglet 1 : Modèle
    with page1:
        # Slider pour choisir le niveau de bruit
        noise_amount = st.slider("Niveau de bruit (salt)", min_value=0.2, max_value=0.5, value=0.2, step=0.01)

        uploaded_files = st.file_uploader("Sélectionnez une ou plusieurs images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        if uploaded_files:
            for idx, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"### Image : {uploaded_file.name}")

                # Chargement et redimensionnement de l'image
                image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
                image_np = np.array(image).astype(np.float32) / 255.0

                # Ajout du bruit sur l'image
                noisy_image = add_salt_noise(image_np, amount=noise_amount)

                # Prédiction
                denoised_image = denoise_image(noisy_image)

                # Affichage
                display_results(image_np, noisy_image, denoised_image, key=idx)
                st.markdown("---")
    with page2:
        # --- 1. Header
        st.markdown("## 📊 Performance du Denoiseur")

        # --- 2. Cartes métriques
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("PSNR (dB)", f"{30.12:.2f}")
        col2.metric("SSIM", f"{0.85:.2f}")
        col3.metric("RMSE", f"{0.035:.3f}")
        col4.metric("Inférence (ms/image)", f"{50:.0f}")
        col5.metric("Taille modèle (Mo)", f"{10:.1f}")
def add_salt_noise_external(image_np, amount=0.01):
    output = np.copy(image_np)
    total_pixels = image_np.shape[0] * image_np.shape[1]
    num_salt = int(amount * total_pixels)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image_np.shape[:2]]
    output[coords[0], coords[1], :] = 1
    return np.clip(output, 0, 1)

def denoise_image_external(image_np, amount=0.01):
    model = tf.keras.models.load_model("../StreamlitApp/model/salt_noise.h5", compile=False)
    model.compile(optimizer="adam", loss="mse")
    resized = tf.image.resize(image_np, (256, 256)).numpy()
    noisy = add_salt_noise_external(resized, amount)
    input_tensor = np.expand_dims(noisy, axis=0)
    denoised = model.predict(input_tensor)[0]
    return np.clip(denoised, 0, 1)
    # Onglet 2 : Performance generales du modèle

