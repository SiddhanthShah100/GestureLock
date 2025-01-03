# GestureLock

GestureLault is a Python-based project that uses hand gestures for encryption and decryption of text. This application employs MediaPipe for hand tracking and gestures, combined with Fernet encryption for securing your data.

## Features
- **Record Gestures**: Save unique hand gestures as a password.
- **Encrypt Data**: Write and encrypt text using the saved gesture as the key.
- **Decrypt Data**: Match the gesture to decrypt and access your text.

## Setup Instructions
1. Clone this repository:
   ```
   git clone https://github.com/SiddhanthShah100/GestureLock.git
   cd GestureLock
   ```

2. Create and activate a virtual environment:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```

3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the program:
   ```
   python main.py
   ```

## Usage Instructions
- **Start the program**:
  ```
  Run the program using python main.py.
  ```

- **Record a Gesture**:
  ```
  Press s to start recording a gesture.
  Perform your gesture and press q to save it.
  ```

- **Encrypt Data**:
  ```
  After recording a gesture, press e to encrypt text.
  A Notepad window will open. Type your text, save it, and close the Notepad.
  ```

- **Decrypt Data**:
  ```
  Press r to record a matching gesture.
  If the gesture matches the saved gesture, a Notepad window will open with the decrypted content.
  ```

- **Exit the Program**:
  ```
  Press esc to stop and exit the program.
  ```

## Additional Notes
```
- Ensure proper lighting conditions and clear visibility of the hand for accurate gesture recognition.
- Keep the gesture consistent during encryption and decryption to avoid mismatches.
- Encrypted files are saved in the encrypted_files directory.
```
