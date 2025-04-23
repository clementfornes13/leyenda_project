import streamlit as st

def show_livrable1():
    st.title("Livrable 1")
    st.write("""
    Ici, vous pouvez mettre tout le contenu spécifique au livrable 1 :
    - Des graphiques
    - Des tableaux
    - Des widgets Streamlit
    """)
    # Ex : un input et un bouton
    name = st.text_input("Entrez votre nom")
    if st.button("Valider"):
        st.success(f"Bonjour, {name} !")