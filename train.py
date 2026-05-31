import cv2
import os
import json
import numpy as np

# Create the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_names = {}

data_dir = "dataset"
current_label = 0

# Loop through each person's folder
for person in sorted(os.listdir(data_dir)):
    person_dir = os.path.join(data_dir, person)

    if not os.path.isdir(person_dir):
        continue

    # Map numeric label to person's name
    label_names[current_label] = person

    # Load all images for this person
    for fname in os.listdir(person_dir):
        path = os.path.join(person_dir, fname)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1

# Train the recognizer
recognizer.train(faces, np.array(labels))

# Save the trained model
recognizer.write("model.yml")

# Save label mappings
with open("labels.json", "w") as f:
    json.dump(label_names, f)

print("Trained on", len(faces), "images.")
print("Model saved as model.yml")
print("Labels saved as labels.json")