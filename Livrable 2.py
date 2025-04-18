import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import io

np.random.seed(42)

# Chargement du modèle
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("modele_unet_salt_256.h5", compile=False)
    model.compile(optimizer="adam", loss="mse")
    return model

model = load_model()

# Fonction pour ajouter du bruit "salt"
def add_salt_noise(image_np, amount=0.05):
    output = np.copy(image_np)
    total_pixels = image_np.shape[0] * image_np.shape[1]
    num_salt = int(amount * total_pixels)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image_np.shape[:2]]
    output[coords[0], coords[1], :] = 1
    return np.clip(output, 0, 1)

# Fonction pour prédire avec le modèle
def denoise_image(noisy_img):
    input_img = np.expand_dims(noisy_img, axis=0)
    denoised = model.predict(input_img)[0]
    return np.clip(denoised, 0., 1.)

# Fonction d'affichage des résultats
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

    # Bouton de téléchargement de l'image débruitée
    denoised_pil = Image.fromarray((denoised * 255).astype(np.uint8))
    buf = io.BytesIO()
    denoised_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="💾 Télécharger l'image débruitée",
        data=byte_im,
        file_name=f"denoised_{key}.png",
        mime="image/png",
        key=f"download_{key}"
    )

# Interface Streamlit
st.title("Débruitage d'image avec U-Net")

# Slider pour choisir le niveau de bruit
noise_amount = st.slider("🔧 Niveau de bruit (salt)", min_value=0.2, max_value=0.5, value=0.2, step=0.01)

uploaded_files = st.file_uploader("Sélectionnez une ou plusieurs images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### Image : {uploaded_file.name}")

        # Chargement et redimensionnement de l'image
        image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
        image_np = np.array(image).astype(np.float32) / 255.0

        # Ajout du bruit salt sur image_np directement
        noisy_image = add_salt_noise(image_np, amount=noise_amount)

        # Prédiction
        denoised_image = denoise_image(noisy_image)

        # Affichage
        display_results(image_np, noisy_image, denoised_image, key=idx)
        st.markdown("---")
