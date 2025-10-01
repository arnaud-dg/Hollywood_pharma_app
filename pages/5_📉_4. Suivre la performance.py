import os
import sys
import base64
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from io import BytesIO
import cv2
import tempfile

# Ajout du chemin racine au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_s3 import list_s3_files, read_image_from_s3, read_file_from_s3

# Configuration de la page
st.set_page_config(page_title="Monitoring", page_icon="📉", layout="wide")

# Charger le logo en base64 (local)
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

# Chargement des styles CSS personnalisés (local)
try:
    with open("assets/css/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Titre de la page
st.title("📉 4. Suivre la performance du modèle")
st.markdown("Module sur les pratiques de validation et de monitoring de l'IA")

# Lister les modèles disponibles pour le groupe sélectionné dans S3
available_models = []
all_models = list_s3_files(prefix="models/", suffix=".h5")
available_models = [os.path.basename(m) for m in all_models if st.session_state.group_choice in os.path.basename(m)]

# Ajouter une option vide en première position
model_options = [""] + available_models

selected_model = st.selectbox(
    "Sélectionner un modèle",
    options=model_options,
    index=0
)

# Bouton pour lancer l'analyse
if st.button("Afficher le suivi de tendance du modèle"):
    if selected_model == "":
        st.warning("⚠️ Veuillez sélectionner un modèle avant de lancer l'analyse.")
        st.stop()
    else:
        import time
        
        # Placeholder pour les messages de progression
        progress_placeholder = st.empty()
        
        # Étape 1 : Téléchargement du modèle
        start_time = time.time()
        progress_placeholder.info("⏳ Étape 1/4 : Téléchargement du modèle depuis S3...")
        
        model_s3_key = f"models/{selected_model}"
        model_bytes = read_file_from_s3(model_s3_key)
        
        if not model_bytes:
            st.error(f"Impossible de charger le modèle {selected_model} depuis S3")
            st.stop()
        
        download_time = time.time() - start_time
        progress_placeholder.success(f"✅ Étape 1/4 terminée : Modèle téléchargé ({download_time:.2f}s)")
        
        # Étape 2 : Chargement du modèle
        start_time = time.time()
        progress_placeholder.info("⏳ Étape 2/4 : Chargement du modèle en mémoire...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file.write(model_bytes)
            temp_model_path = tmp_file.name
        
        try:
            model = load_model(temp_model_path)
        finally:
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
        
        load_time = time.time() - start_time
        progress_placeholder.success(f"✅ Étape 2/4 terminée : Modèle chargé ({load_time:.2f}s)")

        # Étape 3 : Analyse des dossiers
        start_time = time.time()
        progress_placeholder.info("⏳ Étape 3/4 : Listage des dossiers dans S3...")
        
        s3_root = "data/data_maintenance"
        all_files = list_s3_files(prefix=s3_root + "/")
        subfolders = sorted(list(set([f.split("/")[2] for f in all_files if f.count("/") >= 3 and f.split("/")[2].startswith("W")])))
        
        list_time = time.time() - start_time
        progress_placeholder.success(f"✅ Étape 3/4 terminée : {len(subfolders)} dossiers trouvés ({list_time:.2f}s)")

        # Étape 4 : Prédictions
        start_time = time.time()
        progress_placeholder.info(f"⏳ Étape 4/4 : Analyse en cours (0/{len(subfolders)} dossiers traités)...")
        
        class_names = ["Good", "Hole", "Scratch", "Spot"]
        results = []

        for idx, folder in enumerate(subfolders):
            folder_start = time.time()
            progress_placeholder.info(f"⏳ Étape 4/4 : Traitement de {folder} ({idx+1}/{len(subfolders)})...")
            
            folder_prefix = f"{s3_root}/{folder}/"
            folder_images = list_s3_files(prefix=folder_prefix, suffix=".jpg")
            folder_images.extend(list_s3_files(prefix=folder_prefix, suffix=".png"))
            folder_images.extend(list_s3_files(prefix=folder_prefix, suffix=".jpeg"))
            
            correct = 0
            total = 0

            for img_s3_key in folder_images:
                img_name = os.path.basename(img_s3_key)
                true_class = img_name.split("_")[0]
                input_size = (224, 224)

                img_bytes_io = read_image_from_s3(img_s3_key)
                if not img_bytes_io:
                    continue
                
                try:
                    img_bytes = img_bytes_io.read()
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img, input_size)
                    img_array = img_resized.astype('float32') / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    pred = model.predict(img_array, verbose=0)
                    pred_class = class_names[np.argmax(pred)]

                    if pred_class == true_class:
                        correct += 1
                    total += 1
                except Exception as e:
                    continue

            accuracy = (correct / total) * 100 if total > 0 else 0
            results.append({"Dossier": folder, "Accuracy": accuracy})
            
            folder_time = time.time() - folder_start
            progress_placeholder.info(f"⏳ Étape 4/4 : {folder} terminé ({folder_time:.1f}s) - {idx+1}/{len(subfolders)} dossiers")

        prediction_time = time.time() - start_time
        total_time = download_time + load_time + list_time + prediction_time
        progress_placeholder.success(f"✅ Analyse terminée ! Temps total : {total_time:.1f}s (Téléchargement: {download_time:.1f}s, Chargement: {load_time:.1f}s, Listage: {list_time:.1f}s, Prédictions: {prediction_time:.1f}s)")


        # DataFrame
        df = pd.DataFrame(results)

        # Graphique
        fig = px.line(
            df,
            x="Dossier",
            y="Accuracy",
            markers=True,
            title="Carte de contrôle - Suivi de la performance du modèle (Accuracy)",
            labels={"Dossier": "Semaine", "Accuracy": "Accuracy (%)"}
        )
        fig.update_yaxes(range=[50, 100])

        # Étiquettes au-dessus des points
        fig.update_traces(
            text=[f"{val:.1f}%" for val in df["Accuracy"]],
            textposition="top center"
        )

        st.plotly_chart(fig, width='stretch')

        # Affichage des erreurs par dossier
        for folder in subfolders:
            folder_prefix = f"{s3_root}/{folder}/"
            folder_images = list_s3_files(prefix=folder_prefix, suffix=".jpg")
            folder_images.extend(list_s3_files(prefix=folder_prefix, suffix=".png"))
            folder_images.extend(list_s3_files(prefix=folder_prefix, suffix=".jpeg"))
            
            errors = []

            for img_s3_key in folder_images:
                img_name = os.path.basename(img_s3_key)
                true_class = img_name.split("_")[0]

                # Charger l'image depuis S3
                img_bytes_io = read_image_from_s3(img_s3_key)
                if not img_bytes_io:
                    continue
                
                try:
                    img_bytes = img_bytes_io.read()
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img, input_size)
                    img_array = img_resized.astype('float32') / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prédiction
                    pred = model.predict(img_array, verbose=0)
                    pred_class = class_names[np.argmax(pred)]
                    confidence = float(np.max(pred))

                    # Garder uniquement les erreurs
                    if pred_class != true_class:
                        errors.append((img_resized, true_class, pred_class, confidence, img_name))
                except:
                    continue

            # Si erreurs → affichage dans un expander
            if errors:
                with st.expander(f"Erreur de prédiction - {folder}", expanded=False):
                    for i in range(0, len(errors), 4):
                        cols = st.columns(4)
                        for j, col in enumerate(cols):
                            if i + j < len(errors):
                                img_array, true_class, pred_class, confidence, img_name = errors[i + j]
                                with col:
                                    st.image(img_array, width='stretch')
                                    st.markdown(
                                        f"<div style='text-align:center'>"
                                        f"<b>Image :</b> {img_name}<br>"
                                        f"<span style='color:blue'><b>Réel :</b> {true_class}</span><br>"
                                        f"<span style='color:red'><b>Prédit :</b> {pred_class}</span><br>"
                                        f"<b>Probabilité :</b> {confidence:.2f}"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
            else:
                with st.expander(f"Erreur de prédiction - {folder}", expanded=False):
                    st.success("✅ Aucune erreur de prédiction pour ce dossier")