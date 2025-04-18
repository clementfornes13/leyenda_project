# PROJET LEYENDA

## Contexte
TouNum est spécialisé dans la numérisation de documents (textes, images, etc.) pour ses clients, qui souhaitent désormais enrichir leurs services par des outils de Machine Learning. Certains clients possèdent d'importants corpus d'images issues de scans ou de photos, pour lesquels une catégorisation automatique et une génération de légendes seraient des atouts majeurs.

Concevoir une solution automatisée permettant :

1. **De trier** les images (binaire : « photo » vs. « autre ») à l’aide de réseaux de neurones.
2. **De prétraiter** les images (filtrage, denoising, amélioration de la qualité) afin de garantir des entrées optimales.
3. **De générer** automatiquement des légendes descriptives (image captioning) en combinant CNN et RNN.

## Prérequis

- Python >= 3.9
- Bibliothèques :
  - scikit-learn
  - TensorFlow
  - pandas
  - ImageIO
  - NumPy
  - Matplotlib

Installez-les via :
```bash
pip install -r requirements.txt
```

## Installation

```bash
git clone https://github.com/clementfornes13/leyenda_project.git
cd leyenda_project
pip install -e .
```

## Utilisation

### 1. Classification binaire
```bash
Livrable 1.ipynb
```

### 2. Prétraitement d’images
```bash
Livrable 2.ipynb
```

### 3. Captioning d’images
```bash
Livrable 3.ipynb
```

Chaque notebook est autonome et présente :
1. Les objectifs.
2. L’exploration et la préparation des données.
3. Le prototypage du modèle.
4. Les résultats et métriques.
