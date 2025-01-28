import os
import pickle

import mediapipe as mp
import cv2

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory for dataset
DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):  # Skip non-directory files
        continue
    
    print(f"Processing directory: {dir_}")
    
    for img_path in os.listdir(dir_path):
        if img_path == ".DS_Store":  # Skip macOS metadata files
            continue

        img_full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(img_full_path)
        if img is None:  # Skip unreadable images
            print(f"Could not read image: {img_full_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append((x - min(x_)) / (max(x_) - min(x_) if max(x_) - min(x_) != 0 else 1))
                    data_aux.append((y - min(y_)) / (max(y_) - min(y_) if max(y_) - min(y_) != 0 else 1))

                data.append(data_aux)
                labels.append(dir_)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

hands.close()
print("Dataset creation complete. Data saved to 'data.pickle'.")
