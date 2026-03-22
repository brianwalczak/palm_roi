# License: MIT License, Copyright (c) 2026 Brian Walczak. See LICENSE file for details.

import cv2
import numpy as np

PADDING_SIZE = 300 # padding size around image (for hand_padding)
BLACK_THRESHOLD = 0.2 # max black area ratio percentage allowed in ROI

# calculate ROI coordinates based on index and pinky landmarks
def roi_coordinates(image, INDEX_FINGER_MCP, PINKY_MCP, WRIST_MCP=None):
    h, w = image.shape[:2]
    upside_down = False

    # write as coordinates
    l1 = np.array([INDEX_FINGER_MCP.x * w, INDEX_FINGER_MCP.y * h]) # index finger
    l2 = np.array([PINKY_MCP.x * w, PINKY_MCP.y * h]) # pinky finger
    wrist = np.array([WRIST_MCP.x * w, WRIST_MCP.y * h]) if WRIST_MCP is not None else None # wrist (for upside down detection)

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

    # detect upside down orientation if wrist is available
    if wrist is not None:
        wrist_rotated = cv2.transform(np.array([[wrist]], dtype=np.float32), R).reshape(2) # rotate wrist by matrix too
        mcp_mid_y = (l1[1] + l2[1]) / 2 # get midpoint y of fingers
        upside_down = wrist_rotated[1] < mcp_mid_y # wrist above fingers means upside down

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

    # shift ROI up by offset
    if upside_down:
        l1[1] += delta_y
        l2[1] += delta_y
    else:
        l1[1] -= delta_y
        l2[1] -= delta_y
    
    return R, roi_height, l1, l2, upside_down

# calculates ROI, crops, and checks for validity
def calculate_roi(image, INDEX_FINGER_MCP, PINKY_MCP, WRIST_MCP=None, use_padding=True, max_black=BLACK_THRESHOLD):
    if use_padding:
        orig_h, orig_w = image.shape[:2]
        pad = PADDING_SIZE // 2
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        new_h, new_w = image.shape[:2]

        # adjust landmarks for padding
        INDEX_FINGER_MCP.x = (INDEX_FINGER_MCP.x * orig_w + pad) / new_w
        INDEX_FINGER_MCP.y = (INDEX_FINGER_MCP.y * orig_h + pad) / new_h
        PINKY_MCP.x = (PINKY_MCP.x * orig_w + pad) / new_w
        PINKY_MCP.y = (PINKY_MCP.y * orig_h + pad) / new_h

        if WRIST_MCP is not None:
            WRIST_MCP.x = (WRIST_MCP.x * orig_w + pad) / new_w
            WRIST_MCP.y = (WRIST_MCP.y * orig_h + pad) / new_h

    R, roi_height, l1, l2, upside_down = roi_coordinates(image, INDEX_FINGER_MCP, PINKY_MCP, WRIST_MCP) # get coordinates
    h, w = image.shape[:2]
    
    rotated = cv2.warpAffine(image, R, (w, h)) # apply rotation to image

    # define ROI coordinates
    if upside_down:
        ver_start = int(l1[1] - roi_height)
        ver_end = l1[1]
    else:
        ver_start = l1[1]
        ver_end = int(l1[1] + roi_height)
    hor_start = l1[0]
    hor_end = l2[0]

    # ensure coordinates are in correct order (top-left to bottom-right)
    ver_start, ver_end = min(ver_start, ver_end), max(ver_start, ver_end)
    hor_start, hor_end = min(hor_start, hor_end), max(hor_start, hor_end)

    return check_roi(rotated, ver_start, ver_end, hor_start, hor_end, max_black, use_padding) # final checks

# crop the ROI and ensure it's valid
def check_roi(image, ver_start, ver_end, hor_start, hor_end, max_black=BLACK_THRESHOLD, use_padding=True):
    roi = image[ver_start:ver_end, hor_start:hor_end]
    roi_h, roi_w = roi.shape[:2]
    h, w = image.shape[:2]
    
    # check if the ROI crosses into the padding border
    if use_padding:
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