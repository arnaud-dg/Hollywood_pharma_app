# Image TensorFlow GPU officielle (CUDA et cuDNN inclus)
FROM tensorflow/tensorflow:2.9.1-gpu

# Définir le répertoire de travail
WORKDIR /app

# Copier ton projet dans l’image
COPY . /app

# Installer les bibliothèques système nécessaires pour l'affichage et le rendu
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip
RUN pip install --upgrade pip

# Installer les dépendances Python de l'app
RUN pip install -r requirements.txt

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Commande par défaut pour lancer ton app
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
