import streamlit as st

def show_finalpipeline():
    st.title("Pipeline entière")
    st.write("Ici, vous montrez l’enchaînement complet de vos étapes.")
    # Exemple de layout en colonnes pour visualiser un pipeline
    cols = st.columns(4)
    étapes = ["Prétraitement", "Modélisation", "Évaluation", "Déploiement"]
    for col, étape in zip(cols, étapes):
        with col:
            st.header(étape)
            st.write(f"Détails de l’étape **{étape}**.")
            st.button(f"Lancer {étape}")