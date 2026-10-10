"""
Green-channel contrast separation + leaf edge detection, live from OBS Virtual Camera.

What it does, per frame:
    1. Isolate green pixels only (HSV hue band) -- everything else goes to black.
    2. Take each green pixel's brightness (V channel) and INVERT it, then run it
       through the INFERNO colormap:
           - darkest green  (low V)  -> inverted to high  -> INFERNO renders YELLOW
           - lightest green (high V) -> inverted to low   -> INFERNO renders BLACK
           - medium greens fall on INFERNO's orange/red/purple band in between
       This gives every distinct shade of green a distinct, high-contrast color,
       with dark-vs-light intentionally reversed for visual separation.
    3. Detect leaf edges (Canny) restricted to the green regions, overlaid in white.

Trackbars let you tune the green hue/saturation/value range and edge thresholds
live, since outdoor lighting on the rover will vary a lot between runs.

Controls:
    q       - quit
    s       - save current frame (all views) to disk
    trackbars in the "Controls" window - tune green range / edge sensitivity
"""

import cv2
import numpy as np
import os
import time


WINDOW_CONTROLS = "Controls"
WINDOW_ORIGINAL = "Original (OBS)"
WINDOW_MASK = "Green Mask"
WINDOW_RESULT = "Contrast-Separated + Edges"

SAVE_DIR = "captures"


def nothing(_x):
    pass


def create_trackbars():
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, 400, 300)
    cv2.createTrackbar("Hue Min", WINDOW_CONTROLS, 25, 179, nothing)
    cv2.createTrackbar("Hue Max", WINDOW_CONTROLS, 95, 179, nothing)
    cv2.createTrackbar("Sat Min", WINDOW_CONTROLS, 30, 255, nothing)
    cv2.createTrackbar("Val Min", WINDOW_CONTROLS, 20, 255, nothing)
    cv2.createTrackbar("Canny Low", WINDOW_CONTROLS, 40, 255, nothing)
    cv2.createTrackbar("Canny High", WINDOW_CONTROLS, 120, 255, nothing)
    cv2.createTrackbar("Morph Kernel", WINDOW_CONTROLS, 3, 15, nothing)


def read_trackbars():
    return {
        "hue_min": cv2.getTrackbarPos("Hue Min", WINDOW_CONTROLS),
        "hue_max": cv2.getTrackbarPos("Hue Max", WINDOW_CONTROLS),
        "sat_min": cv2.getTrackbarPos("Sat Min", WINDOW_CONTROLS),
        "val_min": cv2.getTrackbarPos("Val Min", WINDOW_CONTROLS),
        "canny_low": cv2.getTrackbarPos("Canny Low", WINDOW_CONTROLS),
        "canny_high": cv2.getTrackbarPos("Canny High", WINDOW_CONTROLS),
        "morph_kernel": max(1, cv2.getTrackbarPos("Morph Kernel", WINDOW_CONTROLS)),
    }


def process_green_contrast(frame_bgr, params, edge_color=(255, 255, 255)):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    green_mask = cv2.inRange(
        hsv,
        (params["hue_min"], params["sat_min"], params["val_min"]),
        (params["hue_max"], 255, 255),
    )

    k = params["morph_kernel"]
    kernel = np.ones((k, k), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Invert brightness: dark green -> high value -> INFERNO yellow
    #                    light green -> low value  -> INFERNO black
    inverted_v = 255 - v
    colored = cv2.applyColorMap(inverted_v, cv2.COLORMAP_INFERNO)

    output = np.zeros_like(frame_bgr)
    output[green_mask > 0] = colored[green_mask > 0]

    # Leaf edges: Canny on grayscale, restricted to (slightly dilated) green regions
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, params["canny_low"], params["canny_high"])
    green_dilated = cv2.dilate(green_mask, kernel, iterations=1)
    edges_on_leaves = cv2.bitwise_and(edges, edges, mask=green_dilated)

    output[edges_on_leaves > 0] = edge_color

    return output, green_mask, edges_on_leaves


def find_obs_camera_index(max_index=8):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Index {i}: frame {frame.shape}")
            cap.release()
    print("Pick the index that matches your OBS Virtual Camera resolution.")


def run(camera_index=0, display_scale=1.0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Run find_obs_camera_index() first to locate the right index."
        )

    create_trackbars()
    os.makedirs(SAVE_DIR, exist_ok=True)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if display_scale != 1.0:
            frame_bgr = cv2.resize(frame_bgr, None, fx=display_scale, fy=display_scale)

        params = read_trackbars()
        result, mask, edges = process_green_contrast(frame_bgr, params)

        cv2.imshow(WINDOW_ORIGINAL, frame_bgr)
        cv2.imshow(WINDOW_MASK, mask)
        cv2.imshow(WINDOW_RESULT, result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = int(time.time())
            cv2.imwrite(f"{SAVE_DIR}/original_{ts}.png", frame_bgr)
            cv2.imwrite(f"{SAVE_DIR}/mask_{ts}.png", mask)
            cv2.imwrite(f"{SAVE_DIR}/result_{ts}.png", result)
            print(f"Saved frame set at {ts}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Step 1: uncomment to find your OBS Virtual Camera's index
    # find_obs_camera_index()

    # Step 2: run with the correct index
    run(camera_index=0)
