# License: MIT License, Copyright (c) 2026 Brian Walczak. See LICENSE file for details.

import cv2
import numpy as np

PADDING_SIZE = 300 # padding size around image (for hand_padding)
BLACK_THRESHOLD = 0.2 # max black area ratio percentage allowed in ROI

# padding to detect close hands and avoid cropping errors when too close
def hand_padding(image, pad=PADDING_SIZE):
    h, w = image.shape[:2]

    # create an empty black canvas
    canvas_h = h + pad
    canvas_w = w + pad
    canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8) # all black

    # calculate offsets to center the image
    top_offset = int(pad / 2)
    bottom_offset = -int(pad / 2)
    left_offset = int(pad / 2)
    right_offset = -int(pad / 2)

    # copy original image into center of canvas
    canvas[top_offset:bottom_offset, left_offset:right_offset, :] = image
    return canvas

# calculate ROI coordinates based on index and pinky landmarks
def roi_coordinates(image, INDEX_FINGER_MCP, PINKY_MCP):
    h, w = image.shape[:2]

    # write as coordinates
    l1 = np.array([INDEX_FINGER_MCP.x * w, INDEX_FINGER_MCP.y * h]) # index finger
    l2 = np.array([PINKY_MCP.x * w, PINKY_MCP.y * h]) # pinky finger

    # ensure left point is first for consistent calculation
    if l1[0] > l2[0]:
        l1, l2 = l2, l1

    angle = np.arctan2((l2 - l1)[1], (l2 - l1)[0]) * 180 / np.pi # calculate angle for rotation

    # normalize angle to avoid upside-down rotations (just in case)
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    # use the midpoint between l1 and l2 as rotation center
    center = ((l1 + l2) / 2).astype(int)
    R = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), angle, 1.0) # calc rotation matrix

    # apply rotation to landmarks
    points = np.array([l1, l2], dtype=np.float32).reshape(-1, 1, 2)
    l1, l2 = cv2.transform(points, R).reshape(-1, 2).astype(int)

    # calculate finger distance w/ small offset
    d = l2[0] - l1[0]
    offset = int(d * 0.05) # 5% offset

    # apply horizontal and vertical offset each side
    l1[0] = int(l1[0] - offset)
    l2[0] = int(l2[0] + offset)
    l1[1] = int(l1[1] - offset)
    l2[1] = int(l2[1] + offset)

    # calculate ROI height before shifting
    roi_height = int(d + (offset * 2))
    delta_y = int(roi_height * 0.15) # 15% of ROI height

    # move l1 and l2 up by delta_y
    l1[1] -= delta_y
    l2[1] -= delta_y
    
    return R, roi_height, l1, l2

# calculates ROI, crops, and checks for validity
def calculate_roi(image, INDEX_FINGER_MCP, PINKY_MCP, max_black=BLACK_THRESHOLD):
    R, roi_height, l1, l2 = roi_coordinates(image, INDEX_FINGER_MCP, PINKY_MCP) # get coordinates
    h, w = image.shape[:2]
    
    rotated = cv2.warpAffine(image, R, (w, h)) # apply rotation to image

    # define ROI coordinates
    ver_start = l1[1]
    ver_end = int(l1[1] + roi_height)
    hor_start = l1[0]
    hor_end = l2[0]

    # ensure coordinates are in correct order (top-left to bottom-right)
    ver_start, ver_end = min(ver_start, ver_end), max(ver_start, ver_end)
    hor_start, hor_end = min(hor_start, hor_end), max(hor_start, hor_end)

    return check_roi(rotated, ver_start, ver_end, hor_start, hor_end, max_black) # final checks

# crop the ROI and ensure it's valid
def check_roi(image, ver_start, ver_end, hor_start, hor_end, max_black=BLACK_THRESHOLD):
    roi = image[ver_start:ver_end, hor_start:hor_end]
    roi_h, roi_w = roi.shape[:2]
    h, w = image.shape[:2]
    
    # check if the ROI crosses into the padding border
    pad_offset = PADDING_SIZE // 2 # padding offset
    margin = 10 # 10 pixel margin for minor mistakes
    if (ver_start < pad_offset - margin or ver_end > h - pad_offset + margin or
        hor_start < pad_offset - margin or hor_end > w - pad_offset + margin):
        return roi, "Error: ROI touches padding border, palm is too close."
    
    # check if ROI is empty (zero width or height for cropping errors)
    if roi_w == 0 or roi_h == 0:
        return roi, "Error: Invalid ROI calculation (empty crop)."

    # check if more than 20% of ROI is fully black
    if max_black is not None and roi.size > 0:
        total_pixels = roi.shape[0] * roi.shape[1] # total number of pixels

        # all channels zero for color, zero for grayscale
        if roi.ndim == 3: # 3 channels
            black_pixels = np.all(roi == 0, axis=2)
        else: # grayscale
            black_pixels = roi == 0
    
        num_black = np.sum(black_pixels) # count black pixels
        black_ratio = num_black / total_pixels if total_pixels > 0 else 0
        if black_ratio > max_black:
            return roi, f"Error: Too much black area in ROI ({black_ratio:.2%})."

    return roi, None