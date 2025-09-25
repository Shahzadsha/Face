import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(("png", "jpg", "jpeg")):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            pil_image = Image.open(path).convert("L")  # grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.Resampling.LANCZOS)
            image_array = np.array(final_image, "uint8")

            faces = face_cascade.detectMultiScale(
                image_array,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                print(f"[WARNING] No face detected in {path}, using whole image instead.")
                # fallback: use full image
                x_train.append(image_array)
                y_labels.append(id_)
            else:
                print(f"[INFO] {path} → Faces found: {len(faces)}")
                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

# Debugging info
print("Total images collected:", len(x_train))
print("Total labels collected:", len(y_labels))
print("Label IDs:", label_ids)

if len(x_train) == 0 or len(y_labels) == 0:
    raise Exception("❌ No training data found! Check your dataset images.")

# Save labels
with open("pickles/face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

# Train recognizer
recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizers/face-trainner.yml")

print("✅ Training complete. Model saved.")
