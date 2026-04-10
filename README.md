# `palm_roi` - A region-of-interest extraction tool 🖐️

<img height="300" src="https://raw.githubusercontent.com/brianwalczak/palm_roi/main/images/cover.png" />

A lightweight palm region-of-interest (ROI) extraction Python library, originally built for use in a palm-vein biometric imaging device. It makes use of OpenCV and hand landmarks (using tools like [MediaPipe](https://github.com/google-ai-edge/mediapipe)) to compute an ROI of a palm in an image, calculate a rotation matrix to straighten it out, and process coordinates.

> The original Raspberry Pi example with a live camera feed has been moved to the `examples/` directory. Check out [the example's README.md](examples/README.md) for details on how to set it up!

## Installation

You can install `palm_roi` easily via pip:

```bash
pip install palm_roi
```

> **Note:** Depending on your setup (e.g., using MediaPipe to get the hand landmarks), your Python version compatibility will be constrained. The `palm_roi` library currently works with Python 3.8+, but MediaPipe currently requires Python <3.13.

## Usage

Here's an example using Mediapipe hands with `palm_roi` to extract hand landmarks and perform ROI computation:

First, download an off-the-shelf model bundle if using Mediapipe hands:
```
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

```python
import palm_roi
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# 1. Provide an image and get hand landmarks from MediaPipe (or another tool)
image = # - your image data - #
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
result = detector.detect(mp_image)

if result.hand_landmarks:
    INDEX_FINGER_MCP = result.hand_landmarks[0][5]
    PINKY_MCP = result.hand_landmarks[0][17]
    WRIST = result.hand_landmarks[0][0]

    # 2. Apply padding, rotation, and cropping to your image automatically.
    output, error = palm_roi.extract(image, INDEX_FINGER_MCP, PINKY_MCP, WRIST)

    # Note: If you do not have or prefer not to provide a wrist landmark, pass `upside_down=True` or `False` directly:
    # output, error = palm_roi.extract(image, INDEX_FINGER_MCP, PINKY_MCP, upside_down=False)

    if not error:
        cv2.imwrite(f"output.png", output) # Save your processed ROI!

    # Alternatively, get the ROI boundaries, rotation matrix, and orientation for manual application.
    R, roi_height, l1, l2, upside_down = palm_roi.get_coords(image, INDEX_FINGER_MCP, PINKY_MCP, WRIST)
    # R, roi_height, l1, l2, upside_down = palm_roi.get_coords(image, INDEX_FINGER_MCP, PINKY_MCP, upside_down=False)
```

The landmark parameters (`INDEX_FINGER_MCP`, `PINKY_MCP`, and `WRIST`) accept standard `(x, y)` coordinate tuples, lists, as well as direct landmark objects from tools like MediaPipe.

**See `examples/main.py` for a full implementation demonstrating real-time camera processing and Mediapipe.**
