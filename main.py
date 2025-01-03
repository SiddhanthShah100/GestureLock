import cv2
import numpy as np
import keyboard
import mediapipe as mp
import os
from tkinter import Tk, Text, Button, END
from cryptography.fernet import Fernet
import base64

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

script_dir = os.path.dirname(os.path.abspath(__file__))
gesture_dir = os.path.join(script_dir, "gestures")
encrypted_dir = os.path.join(script_dir, "encrypted_files")

os.makedirs(gesture_dir, exist_ok=True)
os.makedirs(encrypted_dir, exist_ok=True)

def preprocess_image(frame):
    """Detect hand landmarks and return the cropped hand region."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = int(min(lm.x for lm in hand_landmarks.landmark) * w)
            y_min = int(min(lm.y for lm in hand_landmarks.landmark) * h)
            x_max = int(max(lm.x for lm in hand_landmarks.landmark) * w)
            y_max = int(max(lm.y for lm in hand_landmarks.landmark) * h)

            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            cropped_hand = frame[y_min:y_max, x_min:x_max]

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            return cropped_hand, frame

    return None, frame

def normalize_landmarks(landmarks):
    """Normalize landmarks to remove effects of translation, scale, and rotation."""
    landmarks = np.array(landmarks)

    mean = np.mean(landmarks, axis=0)
    centered = landmarks - mean

    scale = np.sqrt(np.sum(centered**2))
    if scale > 0:
        scaled = centered / scale
    else:
        scaled = centered

    covariance_matrix = np.cov(scaled[:, :2].T)
    eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
    principal_axis = eig_vecs[:, np.argmax(eig_vals)]

    rotation_matrix = np.array([[principal_axis[0], -principal_axis[1]],
                                 [principal_axis[1], principal_axis[0]]])
    aligned = np.dot(scaled[:, :2], rotation_matrix)

    return np.column_stack((aligned, scaled[:, 2])) if scaled.shape[1] > 2 else aligned

def capture_gesture(filename, image_filename):
    """Capture and save a gesture as landmarks and an image."""
    cap = cv2.VideoCapture(0)
    print(f"Start capturing gesture for {filename}. Press 'q' to save and stop.")
    gesture_filename = os.path.join(gesture_dir, filename)
    image_filepath = os.path.join(gesture_dir, image_filename)
    landmarks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame, frame_with_landmarks = preprocess_image(frame)
        if preprocessed_frame is not None:
            display_frame = frame_with_landmarks
            landmarks = extract_landmarks(frame)
        else:
            display_frame = frame

        cv2.imshow("Capture Gesture", display_frame)

        if keyboard.is_pressed('q'):
            if landmarks:
                np.save(gesture_filename, np.array(landmarks))
                cv2.imwrite(image_filepath, display_frame)
                print(f"Gesture saved as {gesture_filename} and image saved as {image_filepath}")
            else:
                print("No hand detected. Try again.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return gesture_filename

def extract_landmarks(frame):
    """Extract and return hand landmarks."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    landmarks = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.append((lm.x, lm.y, lm.z))
    return landmarks

def compare_gestures(gesture1, gesture2):
    """Compare two gestures using normalized and aligned landmarks."""
    landmarks1 = normalize_landmarks(np.load(gesture1))
    landmarks2 = normalize_landmarks(np.load(gesture2))

    if len(landmarks1) != len(landmarks2):
        print("The gestures have different numbers of landmarks.")
        return False

    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
    matching_factor = np.mean(distances < 0.1)

    print(f"Matching factor: {matching_factor:.2f}")
    return matching_factor > 0.65


def decrypt_content(gesture_file):
    """Decrypt the content using the gesture file as the key."""
    with open(gesture_file, 'rb') as img_file:
        img_data = img_file.read()

    hashed_key = base64.urlsafe_b64encode(img_data[:32].ljust(32, b'\0'))
    cipher = Fernet(hashed_key)

    encrypted_filename = os.path.join(encrypted_dir, "encrypted_file.enc")
    with open(encrypted_filename, 'rb') as enc_file:
        encrypted_content = enc_file.read()

    decrypted_content = cipher.decrypt(encrypted_content)
    return decrypted_content.decode()

def open_notepad_and_decrypt(gesture_file):
    """Open a Tkinter notepad to display decrypted content."""
    content = decrypt_content(gesture_file)

    root = Tk()
    root.title("Notepad")

    text = Text(root, wrap='word', font=("Helvetica", 12))
    text.insert(END, content)
    text.pack(expand=True, fill='both')

    close_button = Button(root, text="Close", command=root.destroy)
    close_button.pack()

    root.mainloop()

def open_notepad_and_encrypt(gesture_file):
    """Open a Notepad-like window to enter text and encrypt it."""
    def save_and_encrypt():
        content = text.get("1.0", END).strip()
        if content:
            with open(gesture_file, 'rb') as img_file:
                img_data = img_file.read()

            hashed_key = base64.urlsafe_b64encode(img_data[:32].ljust(32, b'\0'))
            cipher = Fernet(hashed_key)

            encrypted_content = cipher.encrypt(content.encode())

            encrypted_filename = os.path.join(encrypted_dir, "encrypted_file.enc")
            with open(encrypted_filename, 'wb') as enc_file:
                enc_file.write(encrypted_content)

            print(f"Content encrypted and saved as {encrypted_filename}")
        root.destroy()

    root = Tk()
    root.title("Notepad")
    text = Text(root, wrap='word', font=("Helvetica", 12))
    text.pack(expand=True, fill='both')

    save_button = Button(root, text="Save and Encrypt", command=save_and_encrypt)
    save_button.pack()

    root.mainloop()

def main():
    stored_gesture = None

    while True:
        if keyboard.is_pressed('s'):
            print("Start capturing encryption gesture")
            stored_gesture = capture_gesture("gesture.npy", "gesture.png")
            print("Gesture saved!")

        elif keyboard.is_pressed('e'):
            if stored_gesture:
                print("Opening Notepad for encryption...")
                open_notepad_and_encrypt(stored_gesture)

        elif keyboard.is_pressed('r'):
            print("Start capturing password gesture")
            password_gesture = capture_gesture("password_gesture.npy", "password_gesture.png")
            if stored_gesture:
                if compare_gestures(stored_gesture, password_gesture):
                    print("Gesture matched! Opening Notepad with decrypted content...")
                    open_notepad_and_decrypt(stored_gesture)
                else:
                    print("Gesture did not match!")
            else:
                print("No stored gesture to compare with!")

        if keyboard.is_pressed('esc'):
            print("Exiting program.")
            break

if __name__ == "__main__":
    main()
