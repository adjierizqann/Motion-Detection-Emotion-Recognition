# Motion and Emotion Recognition

A compact webcam demonstration that combines frame-difference motion detection,
OpenCV face detection, and DeepFace emotion classification in one real-time
view.

## How it works

1. Consecutive grayscale frames are compared to locate significant motion.
2. OpenCV's bundled Haar cascade locates faces in the current frame.
3. Each face region is passed to DeepFace for dominant-emotion estimation.
4. Motion boxes and emotion labels are rendered locally on the webcam feed.

The application does not upload or persist camera frames.

## Requirements

- Python 3.10 or 3.11
- A webcam accessible to OpenCV
- Desktop access for the OpenCV preview window

```bash
git clone https://github.com/adjierizqan/Motion-Detection-Emotion-Recognition.git
cd Motion-Detection-Emotion-Recognition
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python motion_emotion_detection.py
```

Press `q` to close the application. DeepFace may download model weights on the
first run, so the initial startup can take longer.

## Limitations

- Frame differencing is sensitive to camera movement and lighting changes.
- Emotion labels are model estimates and must not be treated as reliable
  measurements of a person's internal state.
- Performance depends on the TensorFlow backend selected by DeepFace.

GitHub Actions checks Python syntax; webcam inference must be tested locally.
