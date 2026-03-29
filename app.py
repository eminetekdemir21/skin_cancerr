import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request
import gdown

app = Flask(__name__)

# 1. KLASÖR KONTROLÜ
UPLOAD_FOLDER = "static"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = "skin_cancer_model.keras"
FILE_ID = "1lX5X71ncFTlkIohrHdHcrxRu5BCcCRVF"

# 2. MODEL İNDİRME VE YÜKLEME
if not os.path.exists(MODEL_PATH):
    print("Model indiriliyor...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

try:
    # Compile=False online sunucularda hata payını azaltır
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model başarıyla yüklendi!")
except Exception as e:
    print(f"Model yükleme hatası: {e}")

labels = [
    'Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis',
    'Dermatofibroma', 'Melanoma (Tehlikeli)', 'Nevus (Ben)', 'Vascular Lesion'
]

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return "Dosya yüklenmedi"
    
    file = request.files['file']
    if file.filename == '':
        return "Dosya seçilmedi"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Görsel İşleme
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Tahmin
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    prediction = labels[class_idx]

    return render_template("index.html", filename=file.filename, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)