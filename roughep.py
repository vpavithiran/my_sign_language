import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('/Users/pavithiranv/programming/sign_project_important/model75.p ', 'rb'))
model = model_dict['model']

# Set up webcam capture
cap = cv2.VideoCapture(0)

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Label dictionary for hand sign characters
labels_dict = {0: 'A', 1: 'B', 2: 'K', 3: 'L', 4: 'M', 5: 'N', 6: 'O', 7: 'P', 8: 'Q', 9: 'R', 
               10: 'S', 11: 'T', 12: 'C', 13: 'U', 14: 'V', 15: 'W', 16: 'X', 17: 'Y', 18: 'Z', 
               19: 'D', 20: 'E', 21: 'F', 22: 'G', 23: 'H', 24: 'I', 25: 'J'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    # Flip the frame horizontally for the mirror effect
    frame = cv2.flip(frame, 1)

    # Get the frame's width and height
    H, W, _ = frame.shape

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the x and y coordinates of the landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize the coordinates and append to data_aux
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append((x - min(x_)) / (max(x_) - min(x_) if max(x_) - min(x_) != 0 else 1))
                data_aux.append((y - min(y_)) / (max(y_) - min(y_) if max(y_) - min(y_) != 0 else 1))

            # Ensure the input is of shape (1, 42) before prediction
            if len(data_aux) == 42:  # Ensure the data has the correct length (21 landmarks * 2 coordinates)
                prediction = model.predict(np.array([data_aux]))  # Pass the data as a 2D array

                # Get the predicted index with the highest probability
                predicted_index = np.argmax(prediction, axis=1)[0]  # Extract the index of the highest probability
                predicted_character = labels_dict[int(predicted_index)]

                # Set bounding box for the hand sign
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Display bounding box and predicted character on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

            # Clear the temporary lists for the next frame
            x_ = []
            y_ = []
            data_aux = []

    # Display the frame with the predictions
    cv2.imshow('frame', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
