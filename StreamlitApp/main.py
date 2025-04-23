import streamlit as st
from Livrable1 import show_livrable1
from Livrable2 import show_livrable2
from Livrable3 import show_livrable3
from FinalPipeline import show_finalpipeline

st.set_page_config(
    page_title="Mon Application Multi-pages",
    layout="wide",
)

st.sidebar.title("Navigation")
page = st.sidebar.radio( [
    "Livrable 1",
    "Livrable 2",
    "Livrable 3",
    "Pipeline entière"
])

if page == "Livrable 1":
    show_livrable1()
elif page == "Livrable 2":
    show_livrable2()
elif page == "Livrable 3":
    show_livrable3()
elif page == "Pipeline entière":
    show_finalpipeline()