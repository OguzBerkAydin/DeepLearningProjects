import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Modeli yükle
model = load_model('weights2.best.keras')

# Başlık ve açıklama
st.title("Kediler ve Köpekler Sınıflandırma")
st.write("Bir fotoğraf yükleyin ve modelin tahminini görün.")

# Dosya yükleme
uploaded_file = st.file_uploader("Fotoğraf Seçin", type=['jpg', 'jpeg', 'png'])

# Tahmini yap
if uploaded_file is not None:
    # Resmi oku ve ön işlemeden geçir
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Tahmini hesapla
    prediction = model.predict(img_array)
    predicted_class = 'dog' if prediction > 0.5 else 'cat'

    # Sonucu göster
    st.write(f"Tahmin Edilen Sınıf: {predicted_class}")

    # Resim göster (isteğe bağlı)
    st.image(uploaded_file, caption=f"{uploaded_file.name} - Tahmin: {predicted_class}")