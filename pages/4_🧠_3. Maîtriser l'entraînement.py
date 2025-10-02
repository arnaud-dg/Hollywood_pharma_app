import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import tensorflow as tf
from keras.callbacks import Callback
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.metrics import Precision, Recall
from keras.optimizers import Adam
import time
import plotly.graph_objects as go
import cv2
from PIL import Image
import json
from datetime import datetime
import glob
import base64
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import tempfile

# Import des fonctions S3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_s3 import (
    list_s3_files,
    read_image_from_s3,
    write_file_to_s3,
    read_file_from_s3,
    get_bucket_name
)

# Fixer la seed pour la reproductibilité
GROUP_SEEDS = {
    "Grp1": 101,
    "Grp2": 202,
    "Grp3": 303,
    "Grp4": 404,
    "Grp5": 505,
    "Grp6": 606,
    "Grp7": 707
}
DEFAULT_SEED = 42

# Configuration de la page
st.set_page_config(page_title="Entraînement de l'IA", page_icon="🧠", layout="wide")

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def get_base64_of_bin_file(bin_file):
    """Charger le logo en base64."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Charger le logo
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

SEED = GROUP_SEEDS.get(st.session_state.group_choice, DEFAULT_SEED)

import random
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Chargement CSS
try:
    with open("assets/css/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Titre de la page
st.title("🔎 3. Maîtriser l'entraînement du modèle")
st.markdown("""
**Objectifs :**
- Grâce à votre expertise, les anomalies détectées précédemment dans le jeu de données ont toutes été corrigées. Il est maintenant prêt à être utilisé pour entraîner notre IA.
- Trouver les paramètres et hyper-paramètres optimaux assurant une bonne performance du modèle en production.
""")


def save_training_log(train_params, metrics, training_time, model_path=None):
    """Sauvegarder les logs d'entraînement dans S3."""
    timestamp = datetime.now()
    log_id = timestamp.strftime("%Y%m%d_%H%M%S")
    
    log_data = {
        "log_id": log_id,
        "timestamp": timestamp.isoformat(),
        "date": timestamp.strftime("%Y-%m-%d"),
        "heure": timestamp.strftime("%H:%M:%S"),
        "train_ratio": train_params['train_ratio'],
        "batch_size": train_params['batch_size'],
        "learning_rate": train_params['learning_rate'],
        "epochs": train_params['epochs'],
        "penalty_balance": train_params['penalty_balance'],
        "accuracy": metrics['accuracy'],
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "training_time": training_time,
        "model_path": model_path
    }
    
    s3_key = f"logs/training_log_{log_id}.json"
    write_file_to_s3(s3_key, json.dumps(log_data, indent=2))
    return log_id


def load_all_logs():
    """Charger tous les logs depuis S3."""
    logs = []
    log_files = list_s3_files(prefix="logs/", suffix=".json")
    
    for log_file in log_files:
        try:
            content = read_file_from_s3(log_file)
            if content:
                log_data = json.loads(content.decode('utf-8'))
                logs.append(log_data)
        except:
            continue
    
    logs.sort(key=lambda x: x['timestamp'], reverse=True)
    return logs


@st.cache_data
def load_images_from_dataset_augmented(s3_prefix="data/dataset_augmented_full", target_size=(224, 224)):
    """Charger les images depuis S3."""
    images, labels, file_paths = [], [], []
    
    all_files = list_s3_files(prefix=s3_prefix)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        return None, None, None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(image_files)
    
    class_counts = {}
    errors_by_class = {}
    
    for idx, s3_key in enumerate(image_files):
        progress_bar.progress((idx + 1) / total)
        
        parts = s3_key.replace(s3_prefix + "/", "").split("/")
        label = parts[0] if len(parts) > 1 else "Unknown"
        
        img_bytes_io = read_image_from_s3(s3_key)
        if not img_bytes_io:
            errors_by_class[label] = errors_by_class.get(label, 0) + 1
            continue
        
        try:
            img_bytes = img_bytes_io.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                errors_by_class[label] = errors_by_class.get(label, 0) + 1
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, target_size)
            
            images.append(img_resized)
            labels.append(label)
            file_paths.append(s3_key)
            class_counts[label] = class_counts.get(label, 0) + 1
        except Exception as e:
            errors_by_class[label] = errors_by_class.get(label, 0) + 1
            continue
        
    progress_bar.empty()
    status_text.empty()
    
    if not images:
        return None, None, None
    
    X = np.array(images, dtype=np.uint8)
    y = np.array(labels)
    return X, y, file_paths


def balance_dataset(X, y, file_paths, target_size=300, random_state=42):
    """Rééquilibrer le dataset."""
    X_balanced, y_balanced, file_paths_balanced = [], [], []
    classes = np.unique(y)
    
    for c in classes:
        idx = np.where(y == c)[0]
        X_class, y_class, f_class = X[idx], y[idx], np.array(file_paths)[idx]
        
        if len(X_class) > target_size:
            X_res, y_res, f_res = resample(
                X_class, y_class, f_class,
                replace=False,
                n_samples=target_size,
                random_state=random_state
            )
        else:
            X_res, y_res, f_res = resample(
                X_class, y_class, f_class,
                replace=True,
                n_samples=target_size,
                random_state=random_state
            )
        
        X_balanced.append(X_res)
        y_balanced.append(y_res)
        file_paths_balanced.append(f_res)
    
    return (
        np.vstack(X_balanced),
        np.hstack(y_balanced),
        np.hstack(file_paths_balanced)
    )


def get_class_weights(num_classes, penalty_balance):
    """Calculer les poids de classe dynamiques."""
    weights = {}
    base = 1.0
    
    for i in range(num_classes):
        if i == 0:
            weights[i] = base * (0.5 + penalty_balance)
        else:
            weights[i] = base * (1.5 - penalty_balance)
    
    return weights


def recall_m(y_true, y_pred):
    """Métrique de rappel personnalisée."""
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true = tf.cast(y_true, tf.int64)
    tp = tf.reduce_sum(tf.cast(y_true == y_pred_classes, tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true != y_pred_classes, tf.float32))
    return tp / (tp + fn + tf.keras.backend.epsilon())


def precision_m(y_true, y_pred):
    """Métrique de précision personnalisée."""
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true = tf.cast(y_true, tf.int64)
    tp = tf.reduce_sum(tf.cast(y_true == y_pred_classes, tf.float32))
    fp = tf.reduce_sum(tf.cast(y_true != y_pred_classes, tf.float32))
    return tp / (tp + fp + tf.keras.backend.epsilon())


def create_model(learning_rate, input_shape=(224, 224, 3), num_classes=2):
    """Créer le modèle CNN."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', precision_m, recall_m]
    )
    
    return model


class StreamlitProgressBar(Callback):
    """Callback personnalisé pour afficher la progression."""
    
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.gif_placeholder = st.empty()
        
        GIF_PATH = "assets/images/animation_nn.gif"
        if os.path.exists(GIF_PATH):
            with self.gif_placeholder.container():
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.image(GIF_PATH, use_container_width=True)
        
        self.progress_bar = st.progress(0, text="⏳ Entraînement en cours...")
    
    def on_epoch_end(self, epoch, logs=None):
        progress = int(((epoch + 1) / self.total_epochs) * 100)
        self.progress_bar.progress(progress, text=f"Époque {epoch+1}/{self.total_epochs}")
    
    def on_train_end(self, logs=None):
        self.gif_placeholder.empty()


def gradCAM_custom(model, image_array, target_size=(224, 224), colormap=cv2.COLORMAP_JET):
    """Générer une visualisation Grad-CAM."""
    try:
        last_conv_idx = None
        for i, layer in enumerate(reversed(model.layers)):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_idx = len(model.layers) - 1 - i
                break
        
        if last_conv_idx is None:
            raise ValueError("Aucune couche Conv2D trouvée dans le modèle.")
        
        with tf.GradientTape() as tape:
            x = image_array
            conv_output = None
            
            for i, layer in enumerate(model.layers):
                x = layer(x, training=False)
                if i == last_conv_idx:
                    conv_output = x
                    tape.watch(conv_output)
            
            predictions = x
            target_class = tf.argmax(predictions[0])
            loss = predictions[0, target_class]
        
        grads = tape.gradient(loss, conv_output)
        
        if grads is None:
            st.error("Impossible de calculer les gradients")
            return None, None
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        
        heatmap_resized = cv2.resize(heatmap.numpy(), target_size)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        return heatmap_resized, heatmap_colored
    
    except Exception as e:
        st.error(f"Erreur lors de la génération de Grad-CAM: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


def generate_pdf_report(model_name, accuracy, precision, recall, false_rejects, 
                       missed_defects, total_samples, training_time, 
                       fig_loss=None, fig_cm=None, errors_images=None):
    """Générer un rapport PDF."""
    from reportlab.lib import colors
    from reportlab.platypus import Image as RLImage, PageBreak
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20, bottomMargin=20)
    styles = getSampleStyleSheet()
    story = []
    temp_files = []
    
    try:
        model_name = str(model_name)
        accuracy = float(accuracy)
        precision = float(precision)
        recall = float(recall)
        false_rejects = int(false_rejects)
        missed_defects = int(missed_defects)
        total_samples = int(total_samples)
        training_time = float(training_time)
        
        story.append(Paragraph("<b>Rapport d'Entraînement du Modèle</b>", styles["Title"]))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph(f"<b>Modèle :</b> {model_name}", styles["Normal"]))
        story.append(Paragraph(f"<b>Date :</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        story.append(Paragraph(f"<b>Durée d'entraînement :</b> {training_time:.2f} secondes", styles["Normal"]))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("<b>Métriques de Performance</b>", styles["Heading2"]))
        story.append(Spacer(1, 10))
        
        data = [
            ["Métrique", "Valeur"],
            ["Accuracy", "{:.2f}%".format(accuracy * 100)],
            ["Précision", "{:.2f}%".format(precision * 100)],
            ["Recall", "{:.2f}%".format(recall * 100)],
            ["Faux rejets", "{} ({:.2f}%)".format(false_rejects, false_rejects * 100 / total_samples)],
            ["Défauts manqués", "{} ({:.2f}%)".format(missed_defects, missed_defects * 100 / total_samples)],
            ["Total échantillons", "{}".format(total_samples)]
        ]
        
        table = Table(data, colWidths=[250, 200])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F2F2F2')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(table)
        
        if fig_loss is not None or fig_cm is not None:
            story.append(PageBreak())
            story.append(Paragraph("<b>Courbes d'Apprentissage et Matrice de Confusion</b>", styles["Heading2"]))
            story.append(Spacer(1, 10))
            
            if fig_loss is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                tmp.close()
                temp_files.append(tmp.name)
                fig_loss.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                story.append(RLImage(tmp.name, width=450, height=300))
                story.append(Spacer(1, 10))
            
            if fig_cm is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                tmp.close()
                temp_files.append(tmp.name)
                fig_cm.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
                story.append(RLImage(tmp.name, width=350, height=300))
        
        if errors_images and len(errors_images) > 0:
            story.append(PageBreak())
            story.append(Paragraph("<b>Erreurs de Classification</b>", styles["Heading2"]))
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"Nombre total d'erreurs : {len(errors_images)}", styles["Normal"]))
            story.append(Spacer(1, 15))
            
            for i in range(0, min(len(errors_images), 12), 4):
                row_data = []
                for j in range(4):
                    if i + j < len(errors_images):
                        img_array, true_class, pred_class, confidence, filename = errors_images[i + j]
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                        tmp.close()
                        temp_files.append(tmp.name)
                        
                        from PIL import Image as PILImage
                        img_pil = PILImage.fromarray(img_array)
                        img_pil.save(tmp.name, 'PNG')
                        
                        text = f"<font size=8><b>{filename}</b><br/>Réel: {true_class}<br/>Prédit: {pred_class}<br/>Conf: {confidence:.2f}</font>"
                        row_data.append([RLImage(tmp.name, width=100, height=100), Paragraph(text, styles["Normal"])])
                    else:
                        row_data.append(["", ""])
                
                if row_data:
                    error_table = Table([list(zip(*row_data))[0], list(zip(*row_data))[1]], colWidths=[120, 120, 120, 120])
                    error_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ]))
                    story.append(error_table)
                    story.append(Spacer(1, 15))
                    
                    if i > 0 and i % 8 == 0 and i + 4 < len(errors_images):
                        story.append(PageBreak())
        
        doc.build(story)
        pdf = buffer.getvalue()
        buffer.close()
        
        return pdf
    
    finally:
        import time
        time.sleep(0.1)
        for tmp_file in temp_files:
            try:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
            except PermissionError:
                time.sleep(0.2)
                try:
                    if os.path.exists(tmp_file):
                        os.unlink(tmp_file)
                except:
                    pass


# =============== INTERFACE PRINCIPALE ===============

tab1, tab2, tab3 = st.tabs(["Entraînement", "Traçabilité & Logs", "Explicabilité"])

# =============== ONGLET 1: ENTRAÎNEMENT =================
with tab1:
    with st.container(border=True):
        st.subheader("⚙️ Paramètres d'entraînement")
        col1, col2 = st.columns(2)
        
        with col1:
            train_ratio = st.slider(
                "Proportion des données utilisées pour l'entraînement",
                min_value=50,
                max_value=95,
                step=5,
                value=50,
                help="📊 Taille du *train set* relativement au *test set*"
            )
            train_ratio = float(train_ratio / 100)
        
        with col2:
            penalty_balance = st.slider(
                "Balance de pénalités",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Glisser à gauche (0) pour pénaliser les faux négatifs. Glisser à droite (1) pour pénaliser faux positifs."
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0005, 0.001, 0.005, 0.01],
                value=0.001,
                help="📈 Amplitude de correction des erreurs de cycle en cycle"
            )
        
        with col2:
            epochs = st.slider(
                "Époques",
                min_value=5,
                max_value=50,
                step=5,
                value=5,
                help="🔄 Nombre de cycles d'apprentissage"
            )
        
        with col3:
            batch_size = st.select_slider(
                "Batch Size",
                options=[16, 32, 64, 128],
                value=16,
                help="📦 Taille du lot d'images simultanément intégré pour l'apprentissage"
            )
    
    st.markdown("---")
    
    train_button = st.button("🎯 Démarrer un nouvel entraînement", type="primary")
    
    if train_button:
        s3_prefix = "data/dataset_augmented_full"
        expected_classes = ["Good", "Hole", "Scratch", "Spot"]
        
        for cls in expected_classes:
            class_path = f"{s3_prefix}/{cls}/"
            class_files = list_s3_files(prefix=class_path, suffix=".jpg")
            class_files.extend(list_s3_files(prefix=class_path, suffix=".png"))
            class_files.extend(list_s3_files(prefix=class_path, suffix=".jpeg"))
        
        X, y, file_paths = load_images_from_dataset_augmented()
        
        if X is None or y is None:
            st.error("❌ Impossible de charger les données.")
            st.stop()
        
        classes_found = np.unique(y)
        
        if len(classes_found) < 4:
            st.warning(f"⚠️ Attention : Seulement {len(classes_found)} classes trouvées au lieu de 4.")
        
        encoder = LabelEncoder()
        encoder.fit(["Good", "Hole", "Scratch", "Spot"])
        y_encoded = encoder.transform(y)
        num_classes = 4
        
        X = X.astype('float32') / 255.0
        X_bal, y_bal, file_paths_bal = balance_dataset(X, y_encoded, file_paths, target_size=300)
        
        X_train, X_test, y_train, y_test, file_train, file_test = train_test_split(
            X_bal, y_bal, file_paths_bal,
            train_size=train_ratio,
            random_state=SEED,
            stratify=y_bal
        )
        
        model = create_model(learning_rate, input_shape=X_train.shape[1:], num_classes=num_classes)
        class_weight = get_class_weights(num_classes, penalty_balance)
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=0,
            callbacks=[StreamlitProgressBar(epochs)]
        )
        training_time = time.time() - start_time
        
        y_pred = np.argmax(model.predict(X_test), axis=1)
        accuracy = np.mean(y_test == y_pred)
        precision = np.sum((y_test == y_pred) & (y_pred == 1)) / (np.sum(y_pred == 1) + 1e-8)
        recall = np.sum((y_test == y_pred) & (y_test == 1)) / (np.sum(y_test == 1) + 1e-8)
        
        # Sauvegarder le modèle dans S3
        today = datetime.now().strftime("%Y%m%d")
        group = st.session_state.group_choice
        
        existing_models = list_s3_files(prefix=f"models/model_{today}_{group}", suffix=".h5")
        number = len(existing_models) + 1
        model_name = f"model_{today}_{group}_{number}.h5"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            temp_model_path = tmp_file.name
        
        try:
            model.save(temp_model_path)
            
            with open(temp_model_path, 'rb') as f:
                s3_model_key = f"models/{model_name}"
                write_file_to_s3(s3_model_key, f.read())
        finally:
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
        
        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall}
        save_training_log(
            {
                'train_ratio': train_ratio,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'penalty_balance': penalty_balance
            },
            metrics,
            training_time,
            s3_model_key
        )
        
        st.success(f"✅ Entraînement terminé en {training_time:.2f}s - Le modèle généré se nomme **{model_name}** et est sauvegardé dans S3")
        
        st.markdown("---")
        st.subheader("📈 Résultats - graphiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_loss, ax = plt.subplots(2, 1, figsize=(12, 9))
            ax[0].plot(history.history["loss"], label="Train Loss")
            if "val_loss" in history.history:
                ax[0].plot(history.history["val_loss"], label="Val Loss")
            ax[0].set_title("Courbe de Loss")
            ax[0].legend()
            
            ax[1].plot(history.history["accuracy"], label="Train Accuracy")
            if "val_accuracy" in history.history:
                ax[1].plot(history.history["val_accuracy"], label="Val Accuracy")
            ax[1].set_title("Courbe d'Accuracy")
            ax[1].legend()
            
            fig_loss.suptitle("Courbes d'Apprentissage", fontsize=16, weight="bold")
            st.pyplot(fig_loss)
        
        with col2:
            cm = confusion_matrix(y_test, y_pred) 
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
            ax.set_title("Matrice de Confusion", fontsize=12, weight="bold")
            st.pyplot(fig_cm)
            # cm = confusion_matrix(y_test, y_pred)  # tu gardes la fonction sklearn
            # fig_cm, ax = plt.subplots()
            # sns.heatmap(
            #     cm, annot=True, fmt='d', cmap='Purples',
            #     xticklabels=encoder.classes_,
            #     yticklabels=encoder.classes_,
            #     cbar=False   # supprime la barre de gradient
            # )
            # ax.set_title("Matrice de Confusion", fontsize=12, weight="bold")
            # ax.set_xlabel("Prédiction", fontsize=10, weight="bold")
            # ax.set_ylabel("Réel", fontsize=10, weight="bold")
            # st.pyplot(fig_cm)
        
        st.subheader("📊 Métriques & Interprétation des résultats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("**Accuracy**", f"{accuracy * 100:.2f}%")
        
        with col2:
            st.metric("**Precision**", f"{precision * 100:.2f}%")
        
        with col3:
            st.metric("**Recall**", f"{recall * 100:.2f}%")
        
        total_samples = len(y_test)
        class_labels = list(encoder.classes_) if hasattr(encoder, "classes_") else ["Good", "Hole", "Scratch", "Spot"]
        good_class_idx = class_labels.index("Good") if "Good" in class_labels else 0
        defect_class_idx = [i for i, c in enumerate(class_labels) if c != "Good"]
        
        false_rejects = np.sum((y_test == good_class_idx) & (y_pred != good_class_idx))
        missed_defects = np.sum((np.isin(y_test, defect_class_idx)) & (y_pred == good_class_idx))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("**Pourcentage de faux-rejets**", f"{false_rejects * 100 / total_samples:.2f}%")
        
        with col2:
            st.metric("**Pourcentage de défauts manqués**", f"{missed_defects * 100 / total_samples:.2f}%")
        
        with col3:
            st.metric("**Durée de l'entrainement**", f"{training_time:.1f}s")
        
        st.write(f"Sur les **{total_samples} comprimés** :")
        st.write(f"Le pourcentage de bonnes prédictions s'élève à **{(total_samples - false_rejects - missed_defects) * 100 / total_samples:.2f}%**.")
        st.write(f"**{false_rejects} unités conformes** ont été rejetées alors qu'elles étaient en réalité bonnes. "
                f"Le pourcentage de **faux rejets** est donc de **{(false_rejects/total_samples)*100:.2f}%**.")
        st.write(f"**{missed_defects} unités défectueuses** (Hole, Scratch, Spot) ont été classées comme conformes. "
                f"Le pourcentage de **défauts manqués** est donc de **{(missed_defects/total_samples)*100:.2f}%**.")
        
        st.markdown("### 🔎 Visualisation des erreurs de classification")
        
        y_pred_proba = model.predict(X_test)
        errors_idx = np.where(y_test != y_pred)[0]
        
        if len(errors_idx) == 0:
            st.success("✅ Aucune erreur de classification dans ce jeu de test.")
        else:
            st.info(f"Nombre total d'erreurs : {len(errors_idx)}")
            
            for i in range(0, len(errors_idx), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < len(errors_idx):
                        idx = errors_idx[i + j]
                        img = (X_test[idx] * 255).astype("uint8")
                        true_class = class_labels[y_test[idx]]
                        pred_class = class_labels[y_pred[idx]]
                        confidence = np.max(y_pred_proba[idx])
                        file_name = os.path.basename(file_test[idx])
                        
                        with col:
                            st.image(img, use_container_width=True)
                            st.markdown(
                                f"<div style='text-align:center'>"
                                f"<b>Image :</b> {file_name}<br>"
                                f"<span style='color:blue'><b>Réel :</b> {true_class}</span><br>"
                                f"<span style='color:red'><b>Prédit :</b> {pred_class}</span><br>"
                                f"<b>Probabilité :</b> {confidence:.2f}"
                                f"</div>",
                                unsafe_allow_html=True
                            )
        
        st.markdown("### 📄 Générer un rapport PDF")
        
        errors_data = []
        if len(errors_idx) > 0:
            for idx in errors_idx[:12]:
                img_array = (X_test[idx] * 255).astype("uint8")
                true_class = class_labels[y_test[idx]]
                pred_class = class_labels[y_pred[idx]]
                confidence = np.max(y_pred_proba[idx])
                filename = os.path.basename(file_test[idx])
                errors_data.append((img_array, true_class, pred_class, confidence, filename))
        
        accuracy_scalar = float(accuracy)
        precision_scalar = float(precision)
        recall_scalar = float(recall)
        false_rejects_scalar = int(false_rejects)
        missed_defects_scalar = int(missed_defects)
        total_samples_scalar = int(total_samples)
        training_time_scalar = float(training_time)
        
        pdf_data = generate_pdf_report(
            str(model_name),
            accuracy_scalar,
            precision_scalar,
            recall_scalar,
            false_rejects_scalar,
            missed_defects_scalar,
            total_samples_scalar,
            training_time_scalar,
            fig_loss=fig_loss,
            fig_cm=fig_cm,
            errors_images=errors_data
        )
        
        if pdf_data is not None:
            st.download_button(
                label="📥 Télécharger le rapport complet (PDF)",
                data=pdf_data,
                file_name=f"rapport_complet_{model_name.replace('.h5','')}.pdf",
                mime="application/pdf"
            )


# =============== ONGLET 2: LOGS =================
with tab2:
    st.header("📋 Logs")
    
    logs = load_all_logs()
    
    if logs:
        st.dataframe(pd.DataFrame(logs))
    else:
        st.info("Aucun log trouvé dans S3")


# =============== ONGLET 3: XAI =================
with tab3:
    st.header("🔍 Analyse d'explicabilité")
    
    available_models = list_s3_files(prefix="models/", suffix=".h5")
    model_names = [os.path.basename(m) for m in available_models]
    
    if model_names:
        selected_model = st.selectbox("Modèle", model_names)
        
        classes = ["Good", "Hole", "Scratch", "Spot"]
        selected_class = st.selectbox("Classe du comprimé", classes)
        
        s3_class_prefix = f"data/dataset_augmented_full/{selected_class}/"
        class_images = list_s3_files(prefix=s3_class_prefix, suffix=".jpg")
        class_images.extend(list_s3_files(prefix=s3_class_prefix, suffix=".png"))
        
        image_names = [os.path.basename(img) for img in class_images]
        
        if image_names:
            selected_image_name = st.selectbox("Image de test", sorted(image_names))
            
            if st.button("Analyser"):
                selected_s3_key = None
                for img_path in class_images:
                    if os.path.basename(img_path) == selected_image_name:
                        selected_s3_key = img_path
                        break
                
                if selected_s3_key is None:
                    st.error(f"Image '{selected_image_name}' introuvable.")
                else:
                    model_s3_key = f"models/{selected_model}"
                    model_bytes = read_file_from_s3(model_s3_key)
                    
                    if model_bytes:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                            tmp_file.write(model_bytes)
                            temp_model_path = tmp_file.name
                        
                        try:
                            model = load_model(temp_model_path, 
                                             custom_objects={'precision_m': precision_m, 
                                                           'recall_m': recall_m})
                        finally:
                            if os.path.exists(temp_model_path):
                                os.remove(temp_model_path)
                        
                        img_bytes_io = read_image_from_s3(selected_s3_key)
                        
                        if img_bytes_io:
                            img_bytes = img_bytes_io.read()
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img_resized = cv2.resize(img_rgb, (224, 224))
                            img_norm = img_resized.astype("float32") / 255.0
                            img_batch = np.expand_dims(img_norm, axis=0)
                            
                            preds = model.predict(img_batch)
                            pred_class = np.argmax(preds[0])
                            confidence = np.max(preds[0])
                            
                            true_label = selected_class
                            class_mapping = {i: c for i, c in enumerate(classes)}
                            pred_label = class_mapping.get(pred_class, str(pred_class))
                            
                            st.markdown(f"<span style='color:blue'>Réel :</span> {true_label}", 
                                      unsafe_allow_html=True)
                            st.markdown(f"<span style='color:red'>Prédit :</span> {pred_label}", 
                                      unsafe_allow_html=True)
                            st.markdown(f"**Probabilité :** {confidence:.2f}", 
                                      unsafe_allow_html=True)
                            
                            if pred_label == true_label:
                                st.success("**Bonne Prédiction** ✅")
                            else:
                                st.error("**Erreur** ❌")
                            
                            heatmap, heatmap_colored = gradCAM_custom(model, img_batch)
                            
                            if heatmap is not None:
                                overlay = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
                                st.image([img_resized, heatmap_colored, overlay],
                                       caption=["Originale", "Heatmap", "Superposée"],
                                       width=250)
        else:
            st.warning(f"Aucune image trouvée pour la classe '{selected_class}' dans S3.")
    else:
        st.warning("Pas de modèle disponible dans S3")