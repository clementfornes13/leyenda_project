import streamlit as st
from Livrable1 import show_livrable1
from Livrable2 import show_livrable2
from Livrable3 import show_livrable3
from FinalPipeline import show_pipeline

st.set_page_config(
    page_title="Projet IA - Streamlit",
    initial_sidebar_state="collapsed",
    layout="centered",
)

st.sidebar.title("Choix du livrable")
page = st.sidebar.radio(options=[
    "Livrable 1",
    "Livrable 2",
    "Livrable 3",
    "Pipeline entière"
], label="")

if page == "Livrable 1":
    show_livrable1()
elif page == "Livrable 2":
    show_livrable2()
elif page == "Livrable 3":
    show_livrable3()
elif page == "Pipeline entière":
    show_pipeline()