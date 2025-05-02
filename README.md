# 🧠 PROJET LEYENDA

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://leyenda.streamlit.app/)

## 📘 Contexte

**TouNum** est spécialisé dans la **numérisation de documents** (textes, images, etc.) pour ses clients, qui souhaitent désormais enrichir leurs services grâce à des outils de **Machine Learning / Deep Learning**.

Certains clients disposent de **volumineux corpus d’images** issues de scans ou de photos. Pour les valoriser, il devient stratégique de leur appliquer :

- une **catégorisation automatique** (type d'image),
- une **amélioration de qualité**,
- et une **génération de légendes**.

Le projet **Leyenda** vise à concevoir une solution **automatisée** capable de :

1. 📂 **Trier** les images (binaire : « photo » vs. « autre ») à l’aide de **réseaux de neurones**.
2. 🧼 **Prétraiter** les images (filtrage, débruitage, amélioration de la netteté) pour garantir des entrées optimales.
3. ✍️ **Générer automatiquement** des **légendes** via un pipeline **CNN + RNN (Image Captioning)**.

---

## ⚙️ Prérequis

- Python ≥ 3.9
- Bibliothèques :
  - `scikit-learn`
  - `TensorFlow`
  - `pandas`
  - `ImageIO`
  - `NumPy`
  - `Matplotlib`
  - `OpenCV`
  - `Pillow`
  - `tqdm`
  - `scikit-image`
  - `Streamlit`

Installez-les avec :

```bash
pip install -r requirements.txt
```

---

## 📦 Installation

Clonez le projet et installez-le localement :

```bash
git clone https://github.com/clementfornes13/leyenda_project.git
cd leyenda_project
pip install -e .
```

---

## 🚀 Utilisation

### 🔹 1. Classification binaire (photo / autre)

```bash
jupyter notebook jupyters/Livrable 1.ipynb
```

### 🔹 2. Prétraitement d’images

```bash
jupyter notebook jupyters/Livrable 2.ipynb
```

### 🔹 3. Génération de légendes (captioning)

```bash
jupyter notebook jupyters/Livrable 3.ipynb
```

Chaque notebook est **autonome** et contient :

- ✅ Les objectifs
- 📊 L’exploration et préparation des données
- 🧠 Le prototypage du modèle
- 📈 Les résultats et métriques de performance

---

## 🌐 Interface utilisateur (Streamlit)

Une interface a été conçue avec **Streamlit** pour permettre aux utilisateurs de tester le pipeline complet :

```bash
streamlit run StreamlitApp/main.py
```

> 🔧 L’interface permet de charger des images, lancer les modèles, et récupérer la légende générée.

---

## 🧠 Modèles utilisés

- **Classification** : Convolutional Neural Network (CNN)
- **Prétraitement** : filtres classiques + normalisation
- **Captioning** : Extraction CNN + Génération de texte avec LSTM (RNN)

---

## 🗂️ Structure du projet

```bash
leyenda_project/
├── jupyters/                 # Notebooks
│   ├── Livrable 1.ipynb      # Classification
│   ├── Livrable 2.ipynb      # Prétraitement
│   └── Livrable 3.ipynb      # Captioning
├── StreamlitApp/             # Interface utilisateur
│   └── main.py
├── requirements.txt          # Dépendances
├── LICENSE                   # Licence
└── README.md                 # Documentation
```

---

## 👥 Auteurs

Projet développé par :

- 👨‍💻 [Clément FORNES](https://github.com/clementfornes13)
- 👨‍💻 [Aymane HILMI](https://github.com/AymaneHilmi)
- 👨‍💻 [Gabin BOURCE](https://github.com/Sataliicki)
- 👨‍💻 [Jordan LANGLET](https://github.com/JordanLanglet)

