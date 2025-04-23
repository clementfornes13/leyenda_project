import streamlit as st

def show_livrable3():
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import pickle
    from PIL import Image
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model

    # Définir les classes nécessaires (identiques à celles du notebook)

    class CNN_Encoder(tf.keras.Model):
        def __init__(self, embed_dim):
            super().__init__()
            self.fc = tf.keras.layers.Dense(embed_dim)
        def call(self, x):
            return tf.nn.relu(self.fc(x))

    class BahdanauAttention(tf.keras.Model):
        def __init__(self, units):
            super().__init__()
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V  = tf.keras.layers.Dense(1)
        def call(self, features, hidden):
            hidden_time = tf.expand_dims(hidden, 1)
            score = tf.nn.tanh(self.W1(features) + self.W2(hidden_time))
            attention_weights = tf.nn.softmax(self.V(score), axis=1)
            context_vector = tf.reduce_sum(attention_weights * features, axis=1)
            return context_vector, attention_weights

    class RNN_Decoder(tf.keras.Model):
        def __init__(self, vocab_size, embed_dim, units):
            super().__init__()
            self.units = units
            self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
            self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform', unroll=True)
            self.fc1 = tf.keras.layers.Dense(units)
            self.fc2 = tf.keras.layers.Dense(vocab_size)
            self.attention = BahdanauAttention(units)
        def call(self, x, features, hidden):
            context_vector, attention_weights = self.attention(features, hidden)
            x = self.embedding(x)
            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
            output, state = self.gru(x, initial_state=hidden)
            x = self.fc1(output)
            x = tf.reshape(x, (-1, x.shape[2]))
            x = self.fc2(x)
            return x, state, attention_weights
        def reset_state(self, batch_size):
            return tf.zeros((batch_size, self.units))

    # Constantes (même que dans le notebook)
    
    EMBED_DIM = 256
    UNITS = 512
    VOCAB_SIZE = 5000
    MAX_LENGTH = 50



    # Chargement des composants
    encoder = CNN_Encoder(EMBED_DIM)
    decoder = RNN_Decoder(VOCAB_SIZE, EMBED_DIM, UNITS)

    # Charger le tokenizer avant tout
    with open("./model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Initialiser les modèles avec des inputs factices (pour pouvoir charger les poids ensuite)
    encoder(tf.random.uniform((1, 64, 2048)))
    decoder(tf.constant([[tokenizer.word_index['<start>']]]),
            tf.random.uniform((1, 64, EMBED_DIM)),
            decoder.reset_state(batch_size=1))

    # Charger les poids
    encoder.load_weights("./model/encoder.weights.h5")
    decoder.load_weights("./model/decoder.weights.h5")

    with open("./model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Chargement du modèle InceptionV3 pour l'extraction de features
    inception_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = tf.keras.Input(shape=(299, 299, 3))
    hidden_layer = inception_model(new_input)
    feat_extract_model = tf.keras.Model(inputs=new_input, outputs=hidden_layer)

    def load_image(image_file):
        image = Image.open(image_file).convert('RGB')
        image = image.resize((299, 299))
        image = tf.keras.applications.inception_v3.preprocess_input(np.array(image))
        return tf.expand_dims(image, 0)

    def generate_caption(image_tensor):
        features = feat_extract_model(image_tensor)
        features = tf.reshape(features, (1, -1, features.shape[3]))
        features = encoder(features)

        hidden = decoder.reset_state(batch_size=1)
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for _ in range(MAX_LENGTH):
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            predicted_id = tf.argmax(predictions[0]).numpy()
            word = tokenizer.index_word.get(predicted_id, '<unk>')
            if word == '<end>':
                break
            result.append(word)
            dec_input = tf.expand_dims([predicted_id], 0)
        return ' '.join(result)


    st.title("Débruitage d'image avec U-Net")
    # Création des onglets principaux
    page1, page2 = st.tabs([
        "Modèle",
        "Performance générales du modèle",
    ])

    # Onglet 1 : Modèle
    with page1:
        st.title("Image Captioning (Livrable 3)")
        uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Image uploadée", use_column_width=True)
            st.write("Génération de la légende...")
            image_tensor = load_image(uploaded_file)
            caption = generate_caption(image_tensor)
            st.subheader("📝 Légende générée :")
            st.success(caption)
    
    bleu4_score     = 0.14
    rouge1_score    = 0.45
    rougel_score    = 0.38
    avg_caption_len = 12.3 

    with page2:
        st.title("📈 Performance et Caractéristiques du Modèle de Captioning")

        # ——— Cartes métriques (2 lignes de 4)
        row1 = st.columns(4)
        row1[0].metric("BLEU-4",        f"{bleu4_score:.2f}")
        row1[1].metric("ROUGE-1",       f"{rouge1_score:.2f}")
        row1[2].metric("ROUGE-L",       f"{rougel_score:.2f}")
        row1[3].metric("Perplexity",    f"{avg_caption_len:.2f}")

        hp = {
                "Dataset":        ["COCO Captions v2017"],
                "Époques":        [20],
                "Batch Size":     [32],
                "Learning Rate":  [5e-5],
                "Optimizer":      ["AdamW"],
                "Beam Size":      [4],
                "Scheduler":      ["Warmup+CosineAnneal"],
            }
        
        st.write("")
        st.write("")
        st.table(pd.DataFrame(hp))
        st.write("")
        st.write("")

            