import cv2
import mediapipe as mp
import csv
import os
import time

# ================= SETTINGS =================
collecting = False
start_time = 0
collection_duration = 3  # seconds
current_label = None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

DATA_FILE = "alphabet_data.csv"

# =============== CREATE CSV HEADER IF NOT EXISTS ===============
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = []

        # 63 features for Left hand
        for i in range(21):
            header += [f'L_x{i}', f'L_y{i}', f'L_z{i}']

        # 63 features for Right hand
        for i in range(21):
            header += [f'R_x{i}', f'R_y{i}', f'R_z{i}']

        header.append('label')
        writer.writerow(header)

# ================= MAIN LOOP =================
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # ================= KEY INPUT =================
        key = cv2.waitKey(1) & 0xFF

        # Press A-Z to start collecting
        if 65 <= key <= 90 or (97 <= key <= 122):
            current_label = chr(key)
            collecting = True
            start_time = time.time()
            print(f"Collecting for {current_label}")

        # ================= AUTO DATA COLLECTION =================
        if collecting and (time.time() - start_time) <= collection_duration:

            if results.multi_hand_landmarks and current_label is not None:

                # Create fixed 126-length vector (63 per hand)
                full_row = [0] * 126

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):

                    hand_label = handedness.classification[0].label

                    temp = []
                    for lm in hand_landmarks.landmark:
                        temp.extend([lm.x, lm.y, lm.z])

                    # Normalize relative to wrist
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y
                    wrist_z = hand_landmarks.landmark[0].z

                    normalized = []
                    for i in range(0, len(temp), 3):
                        normalized.extend([
                            temp[i] - wrist_x,
                            temp[i+1] - wrist_y,
                            temp[i+2] - wrist_z
                        ])

                    if hand_label == "Left":
                        full_row[0:63] = normalized
                    elif hand_label == "Right":
                        full_row[63:126] = normalized

                full_row.append(current_label)

                with open(DATA_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(full_row)

        elif collecting and (time.time() - start_time) > collection_duration:
            collecting = False
            print("Collection done")

        # ================= DISPLAY INFO =================
        cv2.putText(frame, f'Label: {current_label}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if collecting:
            remaining = round(collection_duration - (time.time() - start_time), 1)
            cv2.putText(frame, f'Recording: {remaining}s', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("ISL Data Collection", frame)

        # ESC to exit
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()