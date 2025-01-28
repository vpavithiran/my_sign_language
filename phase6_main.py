import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3  # Importing pyttsx3 for text-to-speech functionality

# Initialize the speech engine
engine = pyttsx3.init()

# Load the model
model_dict = pickle.load(open('model75.p', 'rb'))
model = model_dict['model']

# Set up webcam capture
cap = cv2.VideoCapture(0)

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Label dictionary for hand sign characters
labels_dict = {
    0: 'A', 1: 'B', 2: 'K', 3: 'L', 4: 'M', 5: 'N', 6: 'O', 7: 'P', 8: 'Q', 9: 'R',
    10: 'S', 11: 'T', 12: 'C', 13: 'U', 14: 'V', 15: 'W', 16: 'X', 17: 'Y', 18: 'Z',
    19: 'D', 20: 'E', 21: 'F', 22: 'G', 23: 'H', 24: 'I', 25: 'J'
}

# Initialize variables
prev_time = 0
curr_time = 0
last_predicted_character = None
last_character_time = 0
displayed_text = ""
last_space_time = 0
space_cooldown = 1  # Cooldown for space bar in seconds

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

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

            if len(data_aux) == 42:
                prediction = model.predict(np.array([data_aux]))
                predicted_index = np.argmax(prediction, axis=1)[0]
                predicted_character = labels_dict[int(predicted_index)]

                time_since_last_character = time.time() - last_character_time
                if predicted_character == last_predicted_character and time_since_last_character < 4:
                    countdown = f"{int(4 - time_since_last_character)}s"
                else:
                    countdown = "Stable"
                    if predicted_character == last_predicted_character:
                        if time_since_last_character >= 4:
                            displayed_text += predicted_character
                            last_character_time = time.time()
                    else:
                        last_predicted_character = predicted_character
                        last_character_time = time.time()

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                            cv2.LINE_AA)
                cv2.putText(frame, countdown, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
                            cv2.LINE_AA)

            x_ = []
            y_ = []
            data_aux = []

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Text: {displayed_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Changed text color to white

    # Display key functionality
    key_info = "Esc: Exit | Space: Add Space | S: Speak Text | C: Clear Text"

    # Calculate text size to ensure it fits on the screen
    text_size = cv2.getTextSize(key_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    # Ensure the key_info text stays within bounds
    x_position = W - text_width - 10  # Add a 10px margin from the right side
    y_position = 20  # Starting from the top

    # If the text width is too large, reduce the font size or place it below the top if it overflows
    if x_position < 0:
        x_position = 10  # Position it to the left if it doesn't fit
        y_position = 50  # Move it down to avoid overlap with the upper part of the frame

    cv2.putText(frame, key_info, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    elapsed_time_since_space = time.time() - last_space_time
    if elapsed_time_since_space < space_cooldown:
        remaining_cooldown = int(space_cooldown - elapsed_time_since_space)
        cv2.putText(frame, f"Space Cooldown: {remaining_cooldown}s", (W // 2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (169, 169, 169), 2, cv2.LINE_AA)  # Changed cooldown text color to gray

    key = cv2.waitKey(1)
    if key == 27:
        cv2.putText(frame, "Exiting... Please wait.", (W // 2 - 100, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Sign Language Translation', frame)
        cv2.waitKey(2000)
        break
    elif key == 32:
        if elapsed_time_since_space >= space_cooldown:
            displayed_text += " "
            last_space_time = time.time()
    elif key == ord('s'):  # Check if 's' key is pressed
        if displayed_text:  # Check if there is any text to speak
            engine.say(displayed_text)  # Speak the displayed text
            engine.runAndWait()  # Wait for the speech to finish
            displayed_text = ""  # Reset the displayed text to an empty string
    elif key == ord('c'):  # Check if 'c' key is pressed
        displayed_text = ""  # Clear the displayed text

    cv2.imshow('Sign Language Translation', frame)

cap.release()
cv2.destroyAllWindows()
