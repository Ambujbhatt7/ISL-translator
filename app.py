import cv2
import mediapipe as mp
import pickle
import tkinter as tk
from PIL import Image, ImageTk

# ================= LOAD MODEL =================
with open("isl_model.pkl", "rb") as f:
    model = pickle.load(f)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= VARIABLES =================
running = False
word = ""
last_prediction = ""
stable_count = 0

# ================= GUI =================
root = tk.Tk()
root.title("ISL Translator")

video_label = tk.Label(root)
video_label.pack()

prediction_label = tk.Label(root, text="Prediction: ", font=("Arial", 20))
prediction_label.pack()

# Output text (word builder)
output_text = tk.StringVar()
output_box = tk.Label(root, textvariable=output_text, font=("Arial", 20))
output_box.pack()

# ================= FUNCTIONS =================
def start_camera():
    global running
    running = True
    update_frame()

def stop_camera():
    global running
    running = False

def clear_text():
    global word
    word = ""
    output_text.set("")

def add_space():
    global word
    word += " "
    output_text.set(word)

def delete_last():
    global word
    word = word[:-1]
    output_text.set(word)

def update_frame():
    global running, word, last_prediction, stable_count, output_text

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = ""

    if results.multi_hand_landmarks:
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
            else:
                full_row[63:126] = normalized

        prediction = model.predict([full_row])[0]

        # ===== WORD BUILDER LOGIC =====
        if prediction == last_prediction:
            stable_count += 1
        else:
            stable_count = 0

        last_prediction = prediction

        if stable_count > 15 and prediction != "":
            word += prediction
            output_text.set(word)
            stable_count = 0
            last_prediction = ""   # RESET to allow same letter again

        # Draw landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Show prediction
    prediction_label.config(text=f"Prediction: {prediction}")

    # Convert frame to Tkinter image
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# ================= BUTTONS =================
start_btn = tk.Button(root, text="Start Camera", command=start_camera)
start_btn.pack()

stop_btn = tk.Button(root, text="Stop Camera", command=stop_camera)
stop_btn.pack()

space_btn = tk.Button(root, text="Space", command=add_space)
space_btn.pack()

delete_btn = tk.Button(root, text="Delete", command=delete_last)
delete_btn.pack()

clear_btn = tk.Button(root, text="Clear", command=clear_text)
clear_btn.pack()

# ================= RUN =================
root.mainloop()

cap.release()
cv2.destroyAllWindows()