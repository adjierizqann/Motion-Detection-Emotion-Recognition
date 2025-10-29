"""Motion detection and emotion recognition using OpenCV and DeepFace.

This script captures frames from a webcam, highlights motion regions, detects faces,
classifies facial emotions with DeepFace, and displays the annotated stream in real time.

Requirements:
    pip install opencv-python deepface
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace


# Minimum contour area (in pixels) to consider as motion. Adjust for your environment.
MOTION_AREA_THRESHOLD: int = 800

# Scaling factor applied to the captured frame to speed up processing.
FRAME_SCALE: float = 0.75


def preprocess_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resize and convert a frame to grayscale for motion detection."""
    if FRAME_SCALE != 1.0:
        frame = cv2.resize(frame, None, fx=FRAME_SCALE, fy=FRAME_SCALE, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return frame, gray


def analyze_emotion(face_roi: np.ndarray) -> str:
    """Return the dominant emotion for a face ROI using DeepFace."""
    if face_roi.size == 0:
        return "unknown"

    rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

    try:
        analysis = DeepFace.analyze(
            rgb_face,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
            prog_bar=False,
            align=False,
        )
    except Exception:
        return "unknown"

    if isinstance(analysis, list):
        analysis = analysis[0]

    return str(analysis.get("dominant_emotion", "unknown"))


def draw_label(frame: np.ndarray, text: str, position: Tuple[int, int]) -> None:
    """Draw a semi-transparent label with text at the specified position."""
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
        raise RuntimeError("Unable to access the webcam. Ensure a webcam is connected and accessible.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")

    prev_gray: Optional[np.ndarray] = None
    last_motion_state = False

    print("Starting motion and emotion detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam. Exiting.")
            break

        processed_frame, gray = preprocess_frame(frame)

        motion_detected = False
        motion_frame = processed_frame.copy()

        if prev_gray is None:
            prev_gray = gray
        else:
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

        # Detect faces on the motion-highlighted frame.
        faces = face_cascade.detectMultiScale(
            cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        for (x, y, w, h) in faces:
            face_roi = motion_frame[y : y + h, x : x + w]
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
