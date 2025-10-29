from __future__ import annotations
import time
from typing import Optional, Tuple
import cv2
import numpy as np
from deepface import DeepFace

MOTION_AREA_THRESHOLD = 800
FRAME_SCALE = 0.75

def preprocess_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if FRAME_SCALE != 1.0:
        frame = cv2.resize(frame, None, fx=FRAME_SCALE, fy=FRAME_SCALE, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return frame, gray

def analyze_emotion(face_roi: np.ndarray) -> str:
    """Return the dominant emotion for a face ROI using DeepFace (OpenCV backend)."""
    try:
        if face_roi.size == 0:
            return "unknown"

        resized_face = cv2.resize(face_roi, (224, 224))

        analysis = DeepFace.analyze(
            resized_face,
            actions=["emotion"],
            enforce_detection=False,   
            detector_backend="opencv"
        )

        if isinstance(analysis, list):
            analysis = analysis[0]

        emotion = analysis.get("dominant_emotion", "unknown")
        return emotion
    except Exception as e:
        print("[DEBUG] Emotion detection error:", e)
        return "unknown"


def draw_label(frame: np.ndarray, text: str, position: Tuple[int, int]) -> None:
    x, y = position
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        frame,
        (x, y - text_height - 6),
        (x + text_width + 6, y),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x + 3, y - 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade")

    prev_gray: Optional[np.ndarray] = None
    last_motion_state = False

    print("Starting motion and emotion detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, gray = preprocess_frame(frame)
        motion_frame = processed_frame.copy()
        motion_detected = False

        # Motion detection
        if prev_gray is not None:
            frame_delta = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < MOTION_AREA_THRESHOLD:
                    continue
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(motion_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        prev_gray = gray

        if motion_detected and not last_motion_state:
            print(f"[{time.strftime('%H:%M:%S')}] Motion detected!")
        last_motion_state = motion_detected

        # Face + emotion detection
        faces = face_cascade.detectMultiScale(
            cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        for (x, y, w, h) in faces:
            face_roi = motion_frame[y:y + h, x:x + w]
            emotion = analyze_emotion(face_roi)
            cv2.rectangle(motion_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            draw_label(motion_frame, emotion, (x, y))

        cv2.imshow("Motion & Emotion Recognition", motion_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
