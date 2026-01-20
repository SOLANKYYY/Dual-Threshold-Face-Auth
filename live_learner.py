# live_learner.py - Final Version with Dual-Threshold Security, Terminal Log, and Sound Alert

import cv2
import numpy as np
import os
from PIL import Image
from playsound import playsound # 🟢 NEW: Import the sound library

# --- CONFIGURATION & INITIAL SETUP ---
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
DATASET_DIR = 'dataset'
TRAINER_FILE = 'trainer/trainer.yml' 
NAMES_FILE = 'names.txt' # File to store ID,Name,Password permanently
SOUND_FILE = 'ring.mp3' # 🟢 NEW: Name of your sound file
TRAINING_SAMPLES = 30 
font = cv2.FONT_HERSHEY_SIMPLEX
face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Ensure required folders exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs('trainer', exist_ok=True) 

# --- FUNCTIONS ---

def get_next_id():
    """Finds the next available unique ID for a new user."""
    max_id = 0
    # Check existing files in the dataset
    for filename in os.listdir(DATASET_DIR):
        if filename.startswith('User.'):
            try:
                id_num = int(filename.split('.')[1])
                if id_num > max_id:
                    max_id = id_num
            except:
                continue
    return max_id + 1

def train_model():
    """Reads all images in the dataset and creates the trainer.yml file."""
    print("\n[INFO] Starting model training...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    image_paths = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.startswith('User.')]
    face_samples = []
    ids = []
    
    for imagePath in image_paths:
        try:
            pil_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(pil_img, 'uint8')
            id_num = int(os.path.split(imagePath)[-1].split(".")[1])
            
            face_samples.append(img_numpy)
            ids.append(id_num)
        except Exception as e:
            print(f"[ERROR] Could not read or process image {imagePath}. Skipping. Error: {e}")
            continue

    if len(face_samples) > 0:
        recognizer.train(face_samples, np.array(ids))
        recognizer.write(TRAINER_FILE)
        print(f"[INFO] Training complete! {len(np.unique(ids))} unique IDs trained. Model saved.")
    else:
        print("[WARNING] Training failed: No valid samples found.")

def load_recognizer_and_names():
    """Loads the latest trained model and maps IDs to names and passwords."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    names = ['Unknown'] # ID 0
    passwords = {} # Dictionary to store ID: Password

    # Read names and passwords from the permanent file 
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2: # Need at least ID and Name
                    try:
                        id_num = int(parts[0])
                        name = parts[1]
                        password = parts[2] if len(parts) > 2 else 'N/A' # Third column is password

                        # Fill the names list up to the required index
                        while len(names) <= id_num:
                            names.append('')
                        
                        names[id_num] = name
                        passwords[id_num] = password
                    except ValueError:
                        continue # Skip bad lines

    # Check if the model exists before reading
    if os.path.exists(TRAINER_FILE):
        try:
            recognizer.read(TRAINER_FILE)
            return recognizer, names, passwords
        except cv2.error as e:
            print(f"[ERROR] Failed to load trainer.yml. Please re-run train_model.py. Error: {e}")
            return None, names, passwords
    else:
        print("[WARNING] No trained model found. Running in detection/capture mode only.")
        return None, names, passwords

# --- STARTUP FIX: Create initial names.txt if it doesn't exist (with OM=04) ---
if not os.path.exists(NAMES_FILE):
    # Set the initial data with the OM password requirement
    initial_data = [
        (1, 'Shah Rukh Khan (DELETED)', 'DELETED'), # Placeholder for original deleted ID 1
        (2, 'OM', '04') # OM's ID (2) is set with password '04'
    ]
    with open(NAMES_FILE, 'w') as f:
        for id_num, name, password in initial_data:
            f.write(f"{id_num},{name},{password}\n")

# Load initial model, names, and passwords
recognizer, names, passwords = load_recognizer_and_names()
cam = cv2.VideoCapture(0)

# --- MAIN LEARNING LOOP ---

print("\nPress 'S' over an unknown face to start the learning process.")
print("Press 'Q' or 'q' to quit.")

# Variable to signal that the loop should break
terminate_app = False 

while True:
    if terminate_app:
        break
        
    ret, img = cam.read()
    if not ret: break
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    faces = face_detector.detectMultiScale(grayImg, scaleFactor=1.2, minNeighbors=5) 
    
    recognized_name = "Unknown"
    box_color = (0, 0, 255) # Red (Unknown)
    best_face_coords = None
    
    # 1. FIND THE BEST CANDIDATE FACE
    if len(faces) > 0:
        x, y, w, h = faces[0] 
        best_face_coords = (x, y, w, h)
        
        # 2. RUN RECOGNITION
        if recognizer:
            id, confidence = recognizer.predict(grayImg[y:y+h, x:x+w])
            
            # 🟢 TIER 1 CHECK: FORGIVING RECOGNITION (Dist < 50) 🟢
            if confidence < 50: 
                # This face is recognized and gets a green box
                recognized_name = names[id] if id < len(names) and names[id] else f"ID {id} (No Name)"
                box_color = (0, 255, 0) # Green (Known)
                
                text_line1 = f"{recognized_name} (Dist: {confidence:.1f})"
                text_line2 = "" # Password line is blank by default

                # 🛑 TIER 2 CHECK: STRICT RECOGNITION (Dist < 40) 🛑
                # MODIFIED: Changed the strict access/termination threshold from < 30 to < 40
                if confidence < 40: 
                    password_text = passwords.get(id, 'N/A')
                    text_line2 = f"PASS: {password_text}"
                    
                    # 🟢 ACTION: PLAY SOUND AND TERMINATE APP 🟢
                    try:
                        playsound(SOUND_FILE, block=False)
                    except Exception as e:
                        print(f"[SOUND ERROR] Could not play sound file '{SOUND_FILE}'. Ensure the file exists and playsound is installed. Error: {e}")
                        
                    print(f"\n[ACCESS GRANTED] HI {recognized_name} SIR YOUR PASS IS {password_text}. HAVE A GOOD DAY")
                    cv2.destroyAllWindows() 
                    terminate_app = True 


                # Display Name/Distance (Line 1 - always shown if < 50)
                cv2.putText(img, text_line1, (x+5, y-25), font, 0.7, (255, 255, 255), 2)
                
                # Display Password (Line 2 - only shown if < 40)
                if text_line2:
                    cv2.putText(img, text_line2, (x+5, y-5), font, 0.7, (0, 255, 255), 2) 
            
            else:
                # UNKNOWN (Dist >= 50)
                text_line1 = f"Unknown (Dist: {confidence:.1f} - Press 'S')"
                cv2.putText(img, text_line1, (x+5, y-25), font, 0.7, (255, 255, 255), 2)
                recognized_name = "Unknown" 
                
        else:
            cv2.putText(img, "Unknown (Press 'S' to Learn)", (x+5, y-25), font, 0.7, (255, 255, 255), 2)


        # 3. DISPLAY BOX (Only drawing one rectangle for simplicity)
        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)


    cv2.imshow("Live Learner - Dual-Threshold Security", img)
    
    # 4. HANDLE KEY PRESSES
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        break
        
    # Start Learning command
    if key == ord('s') or key == ord('S'):
        if best_face_coords and recognized_name == "Unknown":
            
            print("\n-------------------------------------------------")
            print("LIVE LEARNING MODE ACTIVATED!")
            
            new_id = get_next_id()
            
            # Ask for Name AND Password
            new_name = input(f"Enter Name for New User (ID {new_id}): ").strip()
            new_password = input(f"Enter PASSWORD for {new_name} (ID {new_id}): ").strip()
            if not new_name or not new_password: 
                print("Learning cancelled.")
                continue

            # Save the name and password permanently 
            with open(NAMES_FILE, 'a') as f:
                f.write(f"{new_id},{new_name},{new_password}\n")
            
            print(f"Starting sample capture for '{new_name}' (ID {new_id}). Look at the camera.")
            
            x, y, w, h = best_face_coords
            count = 0
            
            while count < TRAINING_SAMPLES:
                ret, capture_img = cam.read()
                if not ret: break

                capture_gray = cv2.cvtColor(capture_img, cv2.COLOR_BGR2GRAY)
                faces_in_capture = face_detector.detectMultiScale(capture_gray, 1.2, 5)

                if len(faces_in_capture) > 0:
                    fx, fy, fw, fh = faces_in_capture[0]
                    count += 1
                    cv2.imwrite(f"{DATASET_DIR}/User.{new_id}.{count}.jpg", capture_gray[fy:fy+fh, fx:fx+fw])
                    
                    cv2.rectangle(capture_img, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
                    cv2.putText(capture_img, f"Capturing: {count}/{TRAINING_SAMPLES}", (20, 50), font, 1, (0, 255, 255), 2)
                    
                cv2.imshow("Live Learner - Capturing...", capture_img)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            
            cv2.destroyWindow("Live Learner - Capturing...")
            print(f"Sample capture complete ({count} samples).")

            # RE-TRAIN THE MODEL with the new data
            train_model()
            
            # RELOAD THE NEW MODEL AND NAMES/PASSWORDS
            recognizer, names, passwords = load_recognizer_and_names()
            print("Model reloaded. Live recognition resuming.")
            print("-------------------------------------------------")


# Cleanup
cam.release()
cv2.destroyAllWindows()