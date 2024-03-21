import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
model_dict = pickle.load(open(os.path.join(current_directory, 'model.p'), 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

label_dict = {0: 'love', 1: 'sad', 2: 'upset', 3: 'excited', 4: 'cold', 5: 'happy', 6: 'please', 7: 'stop', 8: 'please', 9: 'L'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.append(x)
                data_aux.append(y)

                x_.append(x)
                y_.append(y)

            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)

            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)

        # Ensure data_aux contains 84 features (42 for each hand)
        while len(data_aux) < 84:
            data_aux.extend([0, 0])  # Padding with zeros if necessary

        # Debugging: Print extracted hand landmarks for inspection
        print("Data Aux:", data_aux)

        # Make a prediction using the model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_index = int(prediction[0])
        predicted_character = label_dict.get(predicted_index, "Unknown")
        print("Predicted Index:", predicted_index)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
