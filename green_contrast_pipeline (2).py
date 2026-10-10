"""
Green-channel contrast separation + leaf edge detection + disease spot labeling,
live from OBS Virtual Camera.

Pipeline per frame:
    1. Isolate green pixels (HSV hue band) -> green_mask.
    2. Close/dilate green_mask heavily -> leaf_region (bridges over diseased
       patches so the whole leaf silhouette is covered, not just green pixels).
    3. INFERNO contrast-separation on inverted V, masked to green_mask (unchanged
       from before).
    4. Canny edges restricted to green regions, overlaid in white (unchanged).
    5. NEW: inside leaf_region, classify non-green blobs by color/shape:
         - brown / dark necrotic, often blotchy         -> "Early Blight"
         - pale / low-saturation, powdery                -> "Powdery Mildew"
         - anything else non-green with enough area      -> "Unidentified"
       Draw a bounding box + label on both the result view and the original feed.

NOTE: this is a heuristic color/shape classifier, not a trained model. It's meant
for a fast live sanity-check on the rover feed. For confirmed classification, feed
the flagged crop to the Claude vision pipeline already used in the dashboard.

Controls:
    q       - quit
    s       - save current frame (all views) to disk
    d       - toggle disease-detection overlay on/off
    trackbars in the "Controls" window - tune green range / edge sensitivity / spot area
"""

import cv2
import numpy as np
import os
import time


WINDOW_CONTROLS = "Controls"
WINDOW_ORIGINAL = "Original (OBS)"
WINDOW_MASK = "Green Mask"
WINDOW_RESULT = "Contrast-Separated + Edges"
WINDOW_DISEASE = "Disease Spots"

SAVE_DIR = "captures"

# Each abnormal blob is classified ONCE (not per-class-mask), based on its mean
# HSV + area + shape. This avoids one physical spot getting sliced into
# multiple overlapping boxes across classes. Tune thresholds to your lighting.
LABEL_COLORS = {
    "Early Blight": (0, 215, 255),           # amber/gold
    "Leaf Rust": (0, 165, 255),               # orange
    "Late Blight / Black Rot": (0, 0, 139),   # dark red
    "Bacterial Leaf Spot": (255, 0, 255),     # magenta
    "Powdery Mildew": (255, 255, 255),        # white
    "Chlorosis / Yellowing": (0, 255, 255),   # yellow
    "Unidentified": (128, 128, 128),          # gray (should be rare now)
}


def classify_blob(mean_h, mean_s, mean_v, area, circularity):
    """Single classification decision per blob. Order matters: check the most
    distinctive signatures first, fall through to broader brown-spot buckets,
    and only land on 'Unidentified' if nothing else fits."""
    # Powdery mildew: pale, washed-out, low saturation, bright
    if mean_s < 60 and mean_v > 150:
        return "Powdery Mildew"

    # Chlorosis / nutrient yellowing: diffuse, large, saturated yellow
    if 18 <= mean_h <= 35 and mean_s > 60 and mean_v > 120 and area > 3000:
        return "Chlorosis / Yellowing"

    # Leaf rust: vivid orange, brighter & more saturated than typical blight
    if 8 <= mean_h <= 20 and mean_s > 120 and mean_v > 100:
        return "Leaf Rust"

    # Very dark blobs: small + round -> bacterial spot, larger/irregular -> black rot
    if mean_v < 55:
        if area < 400 and circularity > 0.55:
            return "Bacterial Leaf Spot"
        return "Late Blight / Black Rot"

    # Small, round, reddish-brown, low area -> bacterial leaf spot
    if 0 <= mean_h <= 10 and circularity > 0.65 and area < 500:
        return "Bacterial Leaf Spot"

    # Classic mid-brown blotchy necrosis -> early blight
    if 3 <= mean_h <= 25 and 55 <= mean_v <= 150:
        return "Early Blight"

    return "Unidentified"


def nothing(_x):
    pass


def create_trackbars():
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROLS, 400, 360)
    cv2.createTrackbar("Hue Min", WINDOW_CONTROLS, 25, 179, nothing)
    cv2.createTrackbar("Hue Max", WINDOW_CONTROLS, 95, 179, nothing)
    cv2.createTrackbar("Sat Min", WINDOW_CONTROLS, 30, 255, nothing)
    cv2.createTrackbar("Val Min", WINDOW_CONTROLS, 20, 255, nothing)
    cv2.createTrackbar("Canny Low", WINDOW_CONTROLS, 40, 255, nothing)
    cv2.createTrackbar("Canny High", WINDOW_CONTROLS, 120, 255, nothing)
    cv2.createTrackbar("Morph Kernel", WINDOW_CONTROLS, 3, 15, nothing)
    cv2.createTrackbar("Min Spot Area", WINDOW_CONTROLS, 60, 2000, nothing)
    cv2.createTrackbar("Leaf Close Kernel", WINDOW_CONTROLS, 21, 51, nothing)


def read_trackbars():
    return {
        "hue_min": cv2.getTrackbarPos("Hue Min", WINDOW_CONTROLS),
        "hue_max": cv2.getTrackbarPos("Hue Max", WINDOW_CONTROLS),
        "sat_min": cv2.getTrackbarPos("Sat Min", WINDOW_CONTROLS),
        "val_min": cv2.getTrackbarPos("Val Min", WINDOW_CONTROLS),
        "canny_low": cv2.getTrackbarPos("Canny Low", WINDOW_CONTROLS),
        "canny_high": cv2.getTrackbarPos("Canny High", WINDOW_CONTROLS),
        "morph_kernel": max(1, cv2.getTrackbarPos("Morph Kernel", WINDOW_CONTROLS)),
        "min_spot_area": max(1, cv2.getTrackbarPos("Min Spot Area", WINDOW_CONTROLS)),
        "leaf_close_kernel": max(3, cv2.getTrackbarPos("Leaf Close Kernel", WINDOW_CONTROLS) | 1),
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

    return output, green_mask, edges_on_leaves, hsv


def build_leaf_region(green_mask, leaf_close_kernel):
    """Close + dilate the green mask to approximate the full leaf silhouette,
    bridging over diseased patches that break green continuity."""
    close_k = np.ones((leaf_close_kernel, leaf_close_kernel), np.uint8)
    leaf_region = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, close_k)
    leaf_region = cv2.dilate(leaf_region, close_k, iterations=1)
    return leaf_region


def detect_disease_spots(hsv, green_mask, leaf_region, min_area):
    """Find every non-green blob inside the leaf silhouette ONCE, then classify
    each blob individually by its own mean color/brightness/shape. Because each
    physical spot produces exactly one contour -> one label, this can't produce
    the overlapping duplicate boxes that per-class thresholding caused.

    Returns (detections, disease_canvas) where disease_canvas is a color image
    with each classified blob filled in its label color (for the debug view).
    """
    clean_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones((7, 7), np.uint8)

    abnormal_mask = cv2.bitwise_and(leaf_region, cv2.bitwise_not(green_mask))
    abnormal_mask = cv2.morphologyEx(abnormal_mask, cv2.MORPH_OPEN, clean_kernel)
    # Close first so fragments of the same physical spot merge into one blob
    # instead of splintering into several overlapping detections.
    abnormal_mask = cv2.morphologyEx(abnormal_mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(abnormal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_chan, s_chan, v_chan = cv2.split(hsv)
    detections = []
    disease_canvas = np.zeros((*abnormal_mask.shape, 3), dtype=np.uint8)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        blob_mask = np.zeros(abnormal_mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [c], -1, 255, thickness=-1)

        mean_h = cv2.mean(h_chan, mask=blob_mask)[0]
        mean_s = cv2.mean(s_chan, mask=blob_mask)[0]
        mean_v = cv2.mean(v_chan, mask=blob_mask)[0]

        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

        label = classify_blob(mean_h, mean_s, mean_v, area, circularity)
        color = LABEL_COLORS[label]

        x, y, w, h = cv2.boundingRect(c)
        detections.append({
            "label": label,
            "bbox": (x, y, w, h),
            "area": area,
            "color": color,
        })
        disease_canvas[blob_mask > 0] = color

    return detections, disease_canvas


def draw_detections(img, detections):
    out = img.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        color = det["color"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text = f"{det['label']} ({int(det['area'])}px)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(out, text, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


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

    disease_overlay_on = True

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if display_scale != 1.0:
            frame_bgr = cv2.resize(frame_bgr, None, fx=display_scale, fy=display_scale)

        params = read_trackbars()
        result, mask, edges, hsv = process_green_contrast(frame_bgr, params)

        detections = []
        original_view = frame_bgr
        result_view = result

        if disease_overlay_on:
            leaf_region = build_leaf_region(mask, params["leaf_close_kernel"])
            detections, disease_canvas = detect_disease_spots(
                hsv, mask, leaf_region, params["min_spot_area"]
            )
            original_view = draw_detections(frame_bgr, detections)
            result_view = draw_detections(result, detections)
            cv2.imshow(WINDOW_DISEASE, disease_canvas)

            if detections:
                summary = ", ".join(f"{d['label']}@{d['bbox'][:2]}" for d in detections)
                print(f"[frame] {len(detections)} flagged region(s): {summary}")

        cv2.imshow(WINDOW_ORIGINAL, original_view)
        cv2.imshow(WINDOW_MASK, mask)
        cv2.imshow(WINDOW_RESULT, result_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            disease_overlay_on = not disease_overlay_on
            if not disease_overlay_on:
                cv2.destroyWindow(WINDOW_DISEASE)
            print(f"Disease overlay: {'ON' if disease_overlay_on else 'OFF'}")
        elif key == ord("s"):
            ts = int(time.time())
            cv2.imwrite(f"{SAVE_DIR}/original_{ts}.png", original_view)
            cv2.imwrite(f"{SAVE_DIR}/mask_{ts}.png", mask)
            cv2.imwrite(f"{SAVE_DIR}/result_{ts}.png", result_view)
            print(f"Saved frame set at {ts}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Step 1: uncomment to find your OBS Virtual Camera's index
    # find_obs_camera_index()

    # Step 2: run with the correct index
    run(camera_index=0)
