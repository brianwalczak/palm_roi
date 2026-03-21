# License: MIT License, Copyright (c) 2026 Brian Walczak. See LICENSE file for details.
# Example usage with OV9281 monochrome camera, tested on Raspberry Pi Zero 2W.

from libs.roi import roi_coordinates, calculate_roi
from picamera2 import Picamera2
import mediapipe as mp
import cv2
import numpy as np

hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.1, min_tracking_confidence=0.4)

# Camera setup and configuration
def camera_setup():
    picam2 = Picamera2()

    # default config for OV9281 camera, using 640x480 resolution
    camera_config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "YUV420"}
    )

    picam2.configure(camera_config)
    picam2.start()
    return picam2

# Get ROI corner coordinates
def get_roi(frame, INDEX_FINGER_MCP, PINKY_MCP):
    R, roi_height, l1, l2 = roi_coordinates(frame, INDEX_FINGER_MCP, PINKY_MCP) # calculate ROI coordinates and rotation matrix
    R_inv = cv2.invertAffineTransform(R) # get inverse rotation matrix

    # Apply inverse rotation to point
    def apply_rotation(pt):
        pt_h = np.array([pt[0], pt[1], 1.0])
        x, y = np.dot(R_inv, pt_h)[:2]
        return (int(x), int(y))

    # calculate ROI corners and apply inverse rotation
    top_left = apply_rotation(l1)
    top_right = apply_rotation(l2)
    bottom_left = apply_rotation((l1[0], int(l1[1] + roi_height)))
    bottom_right = apply_rotation((l2[0], int(l2[1] + roi_height)))
    return top_left, top_right, bottom_left, bottom_right

# Draw ROI guidance lines on live feed
def draw_guidance(frame, INDEX_FINGER_MCP, PINKY_MCP):
    # create display for colored overlays (OV9281 uses grayscale input)
    display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Draw hand landmarks if available
    if INDEX_FINGER_MCP and PINKY_MCP:
        top_left, top_right, bottom_left, bottom_right = get_roi(frame, INDEX_FINGER_MCP, PINKY_MCP) # get ROI corners

        # draw each line for ROI box
        cv2.line(display, top_left, top_right, (0, 255, 0), 2)
        cv2.line(display, top_right, bottom_right, (0, 255, 0), 2)
        cv2.line(display, bottom_right, bottom_left, (0, 255, 0), 2)
        cv2.line(display, bottom_left, top_left, (0, 255, 0), 2)

    return display

def main():
    picam2 = camera_setup()
    print("palm_roi - Copyright (c) 2026 Brian Walczak, licensed under MIT License.")
    print("This is an example implementation for extracting palm ROI using an OV9281 monochrome camera.")
    print("Press 'q' to quit, 's' to save current frame.")

    try:
        while True:
            raw = picam2.capture_array()
            frame = raw[:480, :640] # y plane only

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)) # convert to RGB for mediapipe
            
            if results.multi_hand_landmarks:
                INDEX_FINGER_MCP = results.multi_hand_landmarks[0].landmark[5] # index finger
                PINKY_MCP = results.multi_hand_landmarks[0].landmark[17] # pinky finger
            else:
                INDEX_FINGER_MCP = None
                PINKY_MCP = None
            
            display = draw_guidance(frame, INDEX_FINGER_MCP, PINKY_MCP) # show ROI overlay on live feed
            cv2.imshow("Output", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if not results.multi_hand_landmarks:
                    print("Error: No hand detected.")
                    continue

                output, error = calculate_roi(frame, INDEX_FINGER_MCP, PINKY_MCP) # get processed ROI

                if error:
                    print(f"Error: {error}")
                else:
                    cv2.imwrite(f"output.png", output) # save processed ROI
                    print("Saved successfully.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
