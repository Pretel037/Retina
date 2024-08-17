import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

# Ruta relativa
model = tf.keras.models.load_model(os.path.join('models', 'cnn.keras'))

# Definir las categorías
categories = ['No_DR', 'Mild', 'Moderate', 'Proliferate_DR', 'Severe']


# Configuraciones de entorno
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Función para preparar la imagen
def prepare_image(img_path, target_size=(150, 150)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalización si tu modelo fue entrenado con imágenes normalizadas
    return img_array

# Función para hacer predicciones
def predict_image(img_path):
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = categories[predicted_index]
    predicted_confidence = predictions[0][predicted_index]
    return predicted_label, predicted_confidence

# Configurar la interfaz de Streamlit
st.title('Clasificación de Retinopatía Diabética')
st.write('Sube hasta 6 imágenes para clasificarlas.')

uploaded_files = st.file_uploader("Elige las imágenes...", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key='files')

if uploaded_files:
    # Asegurarse de que no se suban más de 6 archivos
    if len(uploaded_files) > 6:
        st.warning("Por favor, sube un máximo de 6 imágenes.")
    else:
        results = []

        # Procesar y predecir cada imagen
        for uploaded_file in uploaded_files:
            # Guardar la imagen subida
            with open(f"uploaded_image_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Hacer predicción
            predicted_label, predicted_confidence = predict_image(f"uploaded_image_{uploaded_file.name}")
            results.append({
                "Nombre del archivo": uploaded_file.name,
                "Predicción": predicted_label,
                "Confianza": f"{predicted_confidence:.2f}"
            })

            # Mostrar la imagen subida
            st.image(uploaded_file, caption=f'Imagen: {uploaded_file.name}', use_column_width=True)

        # Mostrar los resultados en una tabla
        df = pd.DataFrame(results)
        st.write("Resultados de la clasificación:")
        st.dataframe(df)
