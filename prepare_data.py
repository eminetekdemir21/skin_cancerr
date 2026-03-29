import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Dosya yolları
IMAGE_DIR1 = r"C:\Users\EXCALIBUR\Desktop\skin_cancer\HAM10000_images_part_1"
IMAGE_DIR2 = r"C:\Users\EXCALIBUR\Desktop\skin_cancer\HAM10000_images_part_2"
CSV_PATH = r"C:\Users\EXCALIBUR\Desktop\skin_cancer\HAM10000_metadata.csv"

# CSV'yi oku
data = pd.read_csv(CSV_PATH)
print("CSV başarıyla okundu, satır sayısı:", len(data))

# Etiketleri sayısal yap
label_dict = {label:i for i,label in enumerate(data['dx'].unique())}
data['label'] = data['dx'].map(label_dict)

# Görselleri oku
images = []
labels = []

missing_files = 0
for idx, row in data.iterrows():
    # Part1 ve Part2'yi kontrol et
    img_path1 = os.path.join(IMAGE_DIR1, row['image_id'] + ".jpg")
    img_path2 = os.path.join(IMAGE_DIR2, row['image_id'] + ".jpg")
    
    if os.path.exists(img_path1):
        img_path = img_path1
    elif os.path.exists(img_path2):
        img_path = img_path2
    else:
        missing_files += 1
        continue
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    images.append(img)
    labels.append(row['label'])

print(f"Görseller okundu: {len(images)}, eksik görsel: {missing_files}")

# Numpy array ve normalizasyon
X = np.array(images)/255.0
y = to_categorical(np.array(labels))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kaydet
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Veri hazırlandı ve kaydedildi!")
