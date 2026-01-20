# train_model.py - FINAL CORRECTED CODE. Run this script once ALL your data is collected.

import cv2
import numpy as np
from PIL import Image
import os

# Configuration
dataset_dir = 'dataset'
trainer_file = 'trainer/trainer.yml'
# Ensure the trainer folder exists
os.makedirs('trainer', exist_ok=True) 

# Initialize the Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]     
    face_samples=[]
    ids = []
    
    for imagePath in image_paths:
        # Check file name validity
        if 'User.' not in imagePath or not (imagePath.endswith('.jpg') or imagePath.endswith('.png')):
            continue
            
        try:
            # 1. Load image and convert to grayscale ('L')
            pil_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(pil_img,'uint8')
            
            # 2. Extract the ID number from the filename
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            
            face_samples.append(img_numpy)
            ids.append(id)
        
        except Exception as e:
            # This catches errors if the image file itself is corrupted or unreadable
            print(f"[ERROR] Could not read image {imagePath}. Skipping. Error: {e}")
            continue

    return face_samples, ids

print ("\n[INFO] Starting model training. Wait ...")

if not os.path.isdir(dataset_dir) or not os.listdir(dataset_dir):
    print("\n[ERROR] The 'dataset' folder is empty or missing. Please ensure you have collected all photos.")
    exit()

faces, ids = get_images_and_labels(dataset_dir)

if len(faces) == 0:
    print("\n[ERROR] No valid face images found in the 'dataset/' folder. Training aborted.")
    exit()

# Check for single ID warning (since you only had 1 ID last time)
if len(np.unique(ids)) < 2:
    print("\n[WARNING] Only one person's ID found. Recognition requires at least two people/IDs (or unknown).")


# Train the recognizer
recognizer.train(faces, np.array(ids))

# Save the model file
recognizer.write(trainer_file) 

# Fix: Use the variable 'trainer_file' instead of 'trainer.yml'
print(f"\n[INFO] Training complete! {len(np.unique(ids))} unique IDs trained. Model saved to {trainer_file}")