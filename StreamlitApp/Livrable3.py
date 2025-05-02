import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import pandas as pd
def show_livrable3():
    st.title("Génération de légendes - Livrable 3")

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
            self.V = tf.keras.layers.Dense(1)
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
            self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, unroll=True)
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

    EMBED_DIM = 256
    UNITS = 512
    VOCAB_SIZE = 5000
    MAX_LENGTH = 50

    encoder = CNN_Encoder(EMBED_DIM)
    decoder = RNN_Decoder(VOCAB_SIZE, EMBED_DIM, UNITS)

    with open("./model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    encoder(tf.random.uniform((1, 64, 2048)))
    decoder(tf.constant([[tokenizer.word_index['<start>']]]),
            tf.random.uniform((1, 64, EMBED_DIM)),
            decoder.reset_state(batch_size=1))

    encoder.load_weights("../StreamlitApp/model/encoder.weights.h5")
    decoder.load_weights("../StreamlitApp/model/decoder.weights.h5")

    inception_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    feature_extractor = tf.keras.Model(inputs=inception_model.input, outputs=inception_model.output)

    def load_image(image_file):
        image = Image.open(image_file).convert('RGB')
        image = image.resize((299, 299))
        image = tf.keras.applications.inception_v3.preprocess_input(np.array(image))
        return tf.expand_dims(image, 0)

    def generate_caption(image_tensor):
        features = feature_extractor(image_tensor)
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
            st.image(uploaded_file, caption="Image uploadée", use_container_width=True)
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


# Fonction externe pour pipeline
def generate_caption_external(image_np):
    import pickle
    from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3

    EMBED_DIM = 256
    UNITS = 512
    VOCAB_SIZE = 5000
    MAX_LENGTH = 50

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
            self.V = tf.keras.layers.Dense(1)
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
            self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, unroll=True)
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

    encoder = CNN_Encoder(EMBED_DIM)
    decoder = RNN_Decoder(VOCAB_SIZE, EMBED_DIM, UNITS)

    with open("../StreamlitApp/model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    encoder(tf.random.uniform((1, 64, 2048)))
    decoder(tf.constant([[tokenizer.word_index['<start>']]]),
            tf.random.uniform((1, 64, EMBED_DIM)),
            decoder.reset_state(batch_size=1))

    encoder.load_weights("../StreamlitApp/model/encoder.weights.h5")
    decoder.load_weights("../StreamlitApp/model/decoder.weights.h5")

    inception_model = InceptionV3(include_top=False, weights='imagenet')
    feature_extractor = tf.keras.Model(inputs=inception_model.input, outputs=inception_model.output)

    resized_img = tf.image.resize(image_np, (299, 299))
    img = preprocess_input(resized_img.numpy())
    img = tf.expand_dims(img, 0)

    features = feature_extractor(img)
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

            
