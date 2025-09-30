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
st.set_page_config(page_title="Glossaire", page_icon="📚", layout="wide")

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

# Dropdown avec session_state
if "group_choice" not in st.session_state:
    st.session_state.group_choice = ""

group_choice = st.sidebar.selectbox(
    "Sélectionner un groupe",
    options=["", "Grp1", "Grp2", "Grp3", "Grp4", "Grp5", "Grp6", "Grp7"],
    index=0 if st.session_state.group_choice == "" else ["", "Grp1", "Grp2", "Grp3", "Grp4", "Grp5", "Grp6", "Grp7"].index(st.session_state.group_choice),
    help="Veuillez choisir un groupe pour afficher le contenu des pages.",
    key="group_choice"
)
st.sidebar.markdown("---")

if st.session_state.group_choice == "":
    st.warning("⚠️ Veuillez sélectionner un groupe dans la barre latérale pour continuer.")
    st.stop()

# Chargement des styles CSS personnalisés
with open("assets/css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Titre de la page
st.title("📚 Glossaire")
st.markdown("Quelques définitions pour mieux comprendre les termes utilisés dans ce projet.")

st.markdown("---")

# Définitions du glossaire
st.markdown(""" 
**🏋️‍♀️ Échantillons d'entraînement** - Ils sont utilisés pour entraîner le modèle à classifier correctement les nouveaux échantillons dans les classes que vous avez créées.

---
            
**✔️ Échantillons de test** - Ils ne sont jamais utilisés pour entraîner le modèle. Ils servent à vérifier les performances du modèle avec de toutes nouvelles données après qu'il ait été entraîné avec les échantillons d'entraînement.

---
            
**📉 Sous-apprentissage** - On parle de sous-apprentissage lorsqu'un modèle classifie mal les **échantillons d'apprentissage** en raison de leur complexité.

---
            
**📈 Surapprentissage** - On parle de surapprentissage lorsqu'un modèle apprend à classifier les **échantillons d'entraînement** avec une telle précision qu'il ne parvient plus à classifier correctement les **échantillons de test**.

---
            
**📅 Époques** - Une époque est une passe complète sur tous les **échantillons d'entraînement** (le modèle les a tous ingérés au moins une fois). Par exemple, si vous avez défini 50 époques, le modèle que vous entraînez fera 50 passes sur l'ensemble de données d'entraînement complet.

---
            
**⚡ Learning Rate (Taux d'apprentissage)** - Le learning rate est un hyperparamètre qui contrôle la vitesse à laquelle le modèle apprend. Une valeur trop élevée peut faire "sauter" le modèle au-dessus de la solution optimale, tandis qu'une valeur trop faible ralentira considérablement l'apprentissage ou le bloquera dans un minimum local.

---
            
**📦 Batch Size (Taille de lot)** - Le batch size détermine le nombre d'échantillons d'entraînement traités simultanément avant de mettre à jour les poids du modèle. Un batch size plus grand stabilise l'apprentissage mais demande plus de mémoire, tandis qu'un batch size plus petit introduit plus de variabilité mais permet un apprentissage plus rapide.

---
            
**📊 Métriques de performance** - Les métriques de performance permettent d'évaluer la qualité des prédictions du modèle. Voici les trois principales métriques :

- **🎯 Accuracy (Précision globale)**
L'**accuracy** mesure le pourcentage de prédictions correctes sur l'ensemble des échantillons. C'est la métrique la plus intuitive : si votre modèle a une accuracy de 85%, cela signifie qu'il classe correctement 85% des échantillons.
- **🔍 Precision (Précision par classe)**
La **precision** mesure, pour une classe donnée, le pourcentage de prédictions correctes parmi toutes les prédictions faites pour cette classe. Elle répond à la question : "Parmi tous les échantillons que le modèle a classés comme positifs, combien le sont réellement ?"
- **📡 Recall (Rappel)**
Le **recall** mesure, pour une classe donnée, le pourcentage d'échantillons correctement identifiés parmi tous les échantillons qui appartiennent réellement à cette classe. Il répond à la question : "Parmi tous les échantillons qui sont réellement positifs, combien le modèle en a-t-il trouvés ?"
""")