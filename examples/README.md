# Example - Raspberry Pi & OV9281

<img height="300" src="../images/cover.png" />

This is an example implementation of `palm_roi` that demonstrates a live camera feed with an ROI overlay and capture, tested on an **OV9281 monochrome camera** connected to a **Raspberry Pi Zero 2W**.

## How It Works

1. **Hand Detection** - Each frame is passed through MediaPipe Hands to locate the index finger MCP, pinky MCP, and (optional) wrist landmarks.
2. **ROI Calculation** - The angle between the index and pinky MCP landmarks is used to compute a rotation matrix to align the palm horizontally. A square ROI is then defined between the landmarks with a small offset and vertical shift to center the palm area.
3. **Orientation Detection** - The wrist landmark is transformed through the same rotation matrix and compared against the midpoint between both MCP landmarks. This determines whether the hand is upside down or not, allowing the ROI to be drawn in the correct direction, at any hand angle.
4. **Guidance Overlay** - On the live feed, the ROI is drawn back (unrotated) as a green bounding box so the user can position their hand.
5. **Capture & Validation** - When you press `s`, the frame is padded (to avoid edge clipping), rotated, and cropped. The output is validated to ensure it isn't empty or filled with too much black area before saving.

## Installation

> **Note:** MediaPipe requires Python <3.13, so you'll need **Raspberry Pi OS Legacy (Debian Bookworm)**, which uses Python 3.11.

From a fresh operating system installation:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install libcap-dev # used by picamera2
# If prompted on the task bar, update packages there as well.

# ONLY FOR OV9281 SETUPS:
sudo nano /boot/firmware/config.txt
# Find "camera-auto-detect=1" and modify it to "camera_auto_detect=0".
# At the end of the file, add "dtoverlay=ov9281".
# Save and exit (Ctrl+X, Y, Enter).
sudo reboot

git clone https://github.com/brianwalczak/palm_roi.git
cd palm_roi
python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r examples/requirements.txt
```

## Usage

```bash
cd examples
python main.py
```

- **`q`** - Quit
- **`s`** - Save the current palm ROI to `output.png`
- **`r`** - Rotate the camera feed 90°