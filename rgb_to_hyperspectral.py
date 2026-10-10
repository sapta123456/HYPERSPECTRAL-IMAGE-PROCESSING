"""
RGB -> Pseudo-Hyperspectral pipeline for the agricultural rover.

Pipeline:
    OBS Virtual Camera (RGB) --> CNN spectral reconstruction --> N-band cube
                                                                --> vegetation mask
                                                                --> false-color / single-band views

IMPORTANT LIMITATION (read before relying on this for disease detection):
An RGB sensor carries no information above ~700nm, so this network can only
reconstruct the VISIBLE spectrum (400-700nm here, 31 bands @ 10nm steps -
the standard ICVL/NTIRE spectral-recovery convention). It cannot recover
true NIR / red-edge reflectance. If your Early Blight / Powdery Mildew
detection relies on NIR-based indices, this will not replace that signal -
it only gives you higher-resolution visible-spectrum bands than a plain
RGB camera. Treat this as a complement to, not a replacement for, your
existing heuristic pseudo-hyperspectral pipeline until an NIR-capable
sensor is added.

TRAINING NOTE:
The model below is architecturally correct but randomly initialized.
Meaningful spectral accuracy requires training on a paired RGB/hyperspectral
dataset (ICVL, CAVE, or NTIRE2020 Spectral Recovery Challenge). A training
script skeleton is included at the bottom. Until trained, treat outputs as
a structural placeholder for pipeline integration/testing, not a
disease-detection-ready spectral estimate.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# 1. Model: lightweight residual CNN, RGB (3ch) -> N spectral bands
#    Architecture follows HSCNN-R / AWAN-style residual spectral reconstruction
# --------------------------------------------------------------------------

BAND_START_NM = 400
BAND_END_NM = 700
NUM_BANDS = 31  # 10nm steps across 400-700nm, matches ICVL/NTIRE convention
BAND_WAVELENGTHS = np.linspace(BAND_START_NM, BAND_END_NM, NUM_BANDS)


class SpectralResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out


class RGB2HSI(nn.Module):
    def __init__(self, out_bands=NUM_BANDS, feat=64, n_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(3, feat, 3, padding=1)
        self.body = nn.Sequential(*[SpectralResBlock(feat) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(feat, out_bands, 3, padding=1)
        self.act = nn.Sigmoid()  # bound output to [0,1] reflectance-like range

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return self.act(x)


# --------------------------------------------------------------------------
# 2. Vegetation masking on the reconstructed cube
#    Uses visible-range proxy index (green-red ratio) since true NDVI needs NIR.
# --------------------------------------------------------------------------

def band_index(nm_target):
    return int(np.argmin(np.abs(BAND_WAVELENGTHS - nm_target)))


def visible_vegetation_mask(cube, green_nm=550, red_nm=670, threshold=1.05):
    """
    cube: (NUM_BANDS, H, W) numpy array
    Returns boolean mask, True where likely vegetation (green > red channel,
    a weak but NIR-free proxy for chlorophyll presence).
    """
    g = cube[band_index(green_nm)]
    r = cube[band_index(red_nm)]
    ratio = g / (r + 1e-6)
    return ratio > threshold


def false_color(cube, r_nm=650, g_nm=550, b_nm=470):
    r = cube[band_index(r_nm)]
    g = cube[band_index(g_nm)]
    b = cube[band_index(b_nm)]
    img = np.stack([r, g, b], axis=-1)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def single_band_view(cube, nm):
    band = cube[band_index(nm)]
    band = (np.clip(band, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(band, cv2.COLORMAP_VIRIDIS)


# --------------------------------------------------------------------------
# 3. Live capture from OBS Virtual Camera
# --------------------------------------------------------------------------

def find_obs_camera_index(max_index=8):
    """
    OBS Virtual Camera doesn't always land on the same index.
    Probe indices and let you visually confirm, or pass CAP_DSHOW on Windows
    to enumerate device names instead of guessing blindly.
    """
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if cv2.os.name == "nt" else 0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Index {i}: frame {frame.shape}")
            cap.release()
    print("Pick the index that matches your OBS Virtual Camera resolution/name.")


def run_pipeline(camera_index=0, device="cpu", display_scale=0.5, current_view_nm=670):
    model = RGB2HSI().to(device)
    model.eval()
    # model.load_state_dict(torch.load("rgb2hsi_weights.pth", map_location=device))
    # ^ uncomment once you have trained weights (see training skeleton below)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Run find_obs_camera_index() first to locate the right index."
        )

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Optional downscale for real-time inference speed
            small = cv2.resize(frame_bgr, None, fx=display_scale, fy=display_scale)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

            cube = model(tensor)[0].cpu().numpy()  # (NUM_BANDS, H, W)

            mask = visible_vegetation_mask(cube)
            fc = false_color(cube)
            band_view = single_band_view(cube, current_view_nm)

            overlay = fc.copy()
            overlay[mask] = (0, 255, 0)

            cv2.imshow("Live RGB (OBS)", small)
            cv2.imshow("Reconstructed False Color", fc)
            cv2.imshow(f"Band @ {current_view_nm}nm", band_view)
            cv2.imshow("Vegetation Overlay", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("["):
                current_view_nm = max(BAND_START_NM, current_view_nm - 10)
            elif key == ord("]"):
                current_view_nm = min(BAND_END_NM, current_view_nm + 10)

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# 4. Training skeleton (run separately, offline, once you have paired data)
# --------------------------------------------------------------------------

def train_skeleton():
    """
    Requires a paired dataset of (RGB image, ground-truth hyperspectral cube).
    Public options: ICVL (icvl.cs.bgu.ac.il), CAVE, or NTIRE2020 Spectral
    Recovery Challenge. Each gives you .mat files with e.g. 31-band cubes
    at 400-700nm matching this script's convention.
    """
    model = RGB2HSI()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()

    # for epoch in range(num_epochs):
    #     for rgb_batch, hsi_gt_batch in dataloader:
    #         pred = model(rgb_batch)
    #         loss = loss_fn(pred, hsi_gt_batch)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    print("Fill in dataloader + training loop once you have a paired dataset.")


if __name__ == "__main__":
    # Step 1: uncomment to find your OBS Virtual Camera's index
    # find_obs_camera_index()

    # Step 2: run the live pipeline (update camera_index once known)
    run_pipeline(camera_index=0, device="cpu")
