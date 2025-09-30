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
st.set_page_config(page_title="Analyse Exploratoire", page_icon="📊", layout="wide")

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
st.title("💼 Présentation du cas pratique")
st.markdown(
    "Pitch sur le cas pratique de Hollywood pharma"
)

st.markdown(""" 

## Contexte
Hollywood Pharma est un laboratoire pharmaceutique ayant développé une nouvelle forme galénique innovante : la **gomme à mâcher** (chewing-gum médicamenteux), contenant un principe actif révolutionnaire nommé provisoirement **« menthol »**.

## Problématique
Les produits sont affectés par des **défauts qualité**. Trois types de défauts prédominent :
1. **Trous** dans la gomme
2. **Points colorés rouges**
3. **Défauts d’enrobage**

## Objectif
Les équipes R&D ont initié un projet d’**intelligence artificielle** pour :
- Détecter automatiquement les défauts sur ligne de production
- Effectuer un tri qualité automatisé

## Votre rôle
Vous intervenez **en tant qu'expert validation IA** :
- Évaluer les risques liés au déploiement du système
- Mettre en place des **verrous de sécurité**
- Garantir les **bonnes pratiques** pour assurer une IA **de confiance**

## Analyse de risque
Les défauts ont des impacts patients différents :
- ✅ **Points rouges** : défaut **esthétique** uniquement, sans impact sur la sécurité ou l’efficacité (pigments non toxiques)
- ⚠️ **Trous et défauts d’enrobage** : peuvent **altérer la cinétique de libération** du principe actif, impactant potentiellement l’**efficacité thérapeutique**

""")

# ### Contexte
# Hollywood Pharma est une entreprise pharmaceutique innovante qui introduit une nouvelle forme galénique, brevetée sous le nom de code « Menthol ». Il s'agit d'un chewing‑gum thérapeutique, pensé pour allier efficacité et praticité.

# ### Enjeux qualité
# En production, plusieurs défauts visuels affectent les gommes : des trous pouvant perturber la libération du principe actif, un enrobage parfois mal réparti, mais aussi certains points rouges. Ces derniers – bien qu'inesthétiques – ne présentent aucun risque toxique ni impact sur l'efficacité.

# ### Solution IA
# Pour remédier à ces imperfections, l'équipe R&D a conçu un système de vision intelligent capable de détecter automatiquement ces anomalies en temps réel et de trier les gommes, garantissant une qualité constante sans ralentir la chaîne de production.

# ### Rôle expert validation
# En tant que spécialiste validation, tu prends en charge l'analyse des risques associés à cette IA. Ton objectif est de mettre en place des verrous de sécurité et des bonnes pratiques robustes, permettant d'obtenir un système non seulement performant, mais également conforme aux exigences réglementaires et digne de confiance.

# ### Gravité des défauts
# Seuls deux types de défauts sont critiques. Les trous et les problèmes d'enrobage peuvent modifier la cinétique de libération du principe actif, menaçant directement son efficacité thérapeutique. En revanche, les points rouges sont des défauts purement esthétiques, sans conséquence sur l'action du médicament.

# ### Objectifs attendus
# L'ambition est simple : garantir une détection sans faille des défauts critiques, assurer une hiérarchisation des anomalies selon leur gravité, instaurer une traçabilité complète de chaque pièce inspectée, et surveiller en continu la performance de l'IA afin d'anticiper toute dérive.

# ### Résultat escompté
# L'initiative vise à renforcer la qualité produit de manière tangible, protéger la sécurité des patients en évitant toute altération de l'efficacité, et surtout assurer une intégration réglementaire sereine grâce à une documentation rigoureuse, un contrôle continu et un pilotage qualité adapté.
