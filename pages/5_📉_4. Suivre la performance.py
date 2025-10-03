import os
import sys
import base64
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from keras.models import load_model
import cv2
import tempfile
import time

# === Streamlit config ===
st.set_page_config(page_title="Monitoring", page_icon="📉", layout="wide")

# === Logo ===
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
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
        unsafe_allow_html=True,
    )
except:
    pass

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

# === CSS optionnel ===
try:
    with open("assets/css/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# === Page title ===
st.title("📉 4. Suivre la performance du modèle")
st.markdown("Module sur les pratiques de validation et de monitoring de l'IA")

# === Définir chemin local des données ===
DATA_ROOT = "data/data_maintenance"

# === Charger modèles depuis models/ ===
models_dir = "models"
available_models = []
if os.path.exists(models_dir):
    available_models = [
        f for f in os.listdir(models_dir)
        if f.endswith(".h5") and st.session_state.group_choice in f
    ]
model_options = [""] + available_models

selected_model = st.selectbox("Sélectionner un modèle", options=model_options, index=0)

# === Bouton ===
if st.button("Afficher le suivi de tendance du modèle"):
    if selected_model == "":
        st.warning("⚠️ Veuillez sélectionner un modèle avant de lancer l'analyse.")
        st.stop()
    else:
        progress_placeholder = st.empty()

        # Étape 1 : Charger modèle local
        start_time = time.time()
        progress_placeholder.info("⏳ Étape 1/3 : Chargement du modèle...")
        model_path = os.path.join(models_dir, selected_model)
        
        if not os.path.exists(model_path):
            st.error(f"Le modèle {model_path} n'existe pas.")
            st.stop()
        
        model = load_model(model_path)
        load_time = time.time() - start_time
        progress_placeholder.success(f"✅ Étape 1/3 terminée : Modèle chargé ({load_time:.2f}s)")

        # Étape 2 : Identifier les sous-dossiers
        start_time = time.time()
        progress_placeholder.info("⏳ Étape 2/3 : Scan des données locales...")
        
        if not os.path.exists(DATA_ROOT):
            st.error(f"Le dossier {DATA_ROOT} n'existe pas.")
            st.stop()
        
        subfolders = sorted(
            [f for f in os.listdir(DATA_ROOT) 
             if os.path.isdir(os.path.join(DATA_ROOT, f)) and f.startswith("W")]
        )
        
        if not subfolders:
            st.warning(f"Aucun dossier commençant par 'W' trouvé dans {DATA_ROOT}")
            st.stop()
        
        list_time = time.time() - start_time
        progress_placeholder.success(f"✅ Étape 2/3 terminée : {len(subfolders)} dossiers trouvés ({list_time:.2f}s)")

        # Étape 3 : Prédictions
        start_time = time.time()
        progress_placeholder.info("⏳ Étape 3/3 : Analyse en cours...")
        class_names = ["Good", "Hole", "Scratch", "Spot"]
        results = []
        errors_all = {}

        input_size = (224, 224)

        for idx, folder in enumerate(subfolders):
            folder_path = os.path.join(DATA_ROOT, folder)
            folder_images = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            X_batch, y_true, img_names, img_resized_list = [], [], [], []
            for img_path in folder_images:
                img_name = os.path.basename(img_path)
                true_class = img_name.split("_")[0]

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img, input_size)
                    X_batch.append(img_resized.astype("float32") / 255.0)
                    y_true.append(true_class)
                    img_names.append(img_name)
                    img_resized_list.append(img_resized)
                except Exception as e:
                    continue

            if len(X_batch) > 0:
                X_batch = np.stack(X_batch, axis=0)
                preds = model.predict(X_batch, batch_size=32, verbose=0)
                y_pred = [class_names[np.argmax(p)] for p in preds]
                confidences = [float(np.max(p)) for p in preds]

                # Accuracy
                correct = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp])
                total = len(y_true)
                accuracy = (correct / total) * 100 if total > 0 else 0
                results.append({"Dossier": folder, "Accuracy": accuracy})

                # Sauvegarde erreurs
                errors = [
                    (img_resized_list[i], y_true[i], y_pred[i], confidences[i], img_names[i])
                    for i in range(len(y_true))
                    if y_pred[i] != y_true[i]
                ]
                if errors:
                    errors_all[folder] = errors

            progress_placeholder.info(f"⏳ Étape 3/3 : {folder} traité ({idx+1}/{len(subfolders)})")

        prediction_time = time.time() - start_time
        total_time = load_time + list_time + prediction_time
        progress_placeholder.success(f"✅ Analyse terminée ! Temps total : {total_time:.1f}s")

        # === Résultats globaux ===
        df = pd.DataFrame(results)
        if not df.empty:
            fig = px.line(
                df,
                x="Dossier",
                y="Accuracy",
                markers=True,
                title="Carte de contrôle - Suivi de la performance du modèle (Accuracy)",
                labels={"Dossier": "Semaine", "Accuracy": "Accuracy (%)"},
            )
            fig.update_yaxes(range=[50, 100])
            fig.update_traces(
                text=[f"{val:.1f}%" for val in df["Accuracy"]],
                textposition="top center",
            )
            st.plotly_chart(fig, use_container_width=True)

        # === Détails erreurs ===
        if errors_all:
            for folder, errors in errors_all.items():
                with st.expander(f"Erreur de prédiction - {folder}", expanded=False):
                    for i in range(0, len(errors), 4):
                        cols = st.columns(4)
                        for j, col in enumerate(cols):
                            if i + j < len(errors):
                                img_array, true_class, pred_class, confidence, img_name = errors[i + j]
                                with col:
                                    st.image(img_array, use_container_width=True)
                                    st.markdown(
                                        f"<div style='text-align:center'>"
                                        f"<b>Image :</b> {img_name}<br>"
                                        f"<span style='color:blue'><b>Réel :</b> {true_class}</span><br>"
                                        f"<span style='color:red'><b>Prédit :</b> {pred_class}</span><br>"
                                        f"<b>Probabilité :</b> {confidence:.2f}"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )
        else:
            st.success("✅ Aucune erreur de prédiction détectée dans les données analysées.")