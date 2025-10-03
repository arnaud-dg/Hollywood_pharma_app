import os
import sys
import base64
import pandas as pd
import plotly.express as px
import streamlit as st

# Ajout du chemin racine au path pour pouvoir importer utils et config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import load_data

# Configuration de la page
st.set_page_config(page_title="Visualisations Avancées", page_icon="📈", layout="wide")

# Charger le logo en base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64_of_bin_file("assets/images/a3p.png")

st.markdown(
    f"""
    <style>
        [data-testid="stSidebarNav"]::before {{
            content: "";
            display: block;
            margin: 0 auto 20px auto;
            height: 200px;
            width: 250px;
            background-image: url("data:image/png;base64,{logo_base64}");
            background-repeat: no-repeat;
            background-size: contain;
            background-position: center;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Initialisation et gestion du groupe
if "group_choice" not in st.session_state:
    st.session_state.group_choice = ""

# Récupérer l'index actuel
current_index = 0
options = ["", "Grp1", "Grp2", "Grp3", "Grp4", "Grp5", "Grp6", "Grp7"]
if st.session_state.group_choice in options:
    current_index = options.index(st.session_state.group_choice)

# Selectbox qui met à jour automatiquement
st.session_state.group_choice = st.sidebar.selectbox(
    "Sélectionner un groupe",
    options=options,
    index=current_index,
    help="Veuillez choisir un groupe pour afficher le contenu des pages."
)
st.sidebar.markdown("---")

if st.session_state.group_choice == "":
    st.warning("⚠️ Veuillez sélectionner un groupe dans la barre latérale pour continuer.")
    st.stop()

# Chargement des styles CSS personnalisés
with open("assets/css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Titre de la page
st.title("⚠️ 1. Evaluer la criticité et identifier les risques projets")
st.info("Ce module n'est pas traité de façon interactive dans cette application, mais en présentiel.")


