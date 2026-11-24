import cv2
import numpy as np


def resize_frame(frame, target_width=640):
    """Resizes frame proportionally to target_width."""
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    ratio = target_width / float(w)
    new_height = int(h * ratio)
    return cv2.resize(frame, (target_width, new_height), interpolation=cv2.INTER_AREA)


def apply_gaussian_blur(frame, kernel_size=(5, 5)):
    """Reduces high-frequency sensor noise common in night CCTV footage."""
    return cv2.GaussianBlur(frame, kernel_size, 0)


def apply_gamma_correction(frame, gamma=1.8):
    """
    Applies non-linear gamma correction using a fast Look-Up Table (LUT).
    gamma > 1.0 brightens dark regions; gamma < 1.0 darkens bright regions.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    return cv2.LUT(frame, table)


def apply_clahe(frame, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the L-channel in LAB color space to enhance local contrast without shifting colors.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)

    merged_lab = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def compute_image_metrics(frame):
    """
    Computes mean brightness and RMS contrast (standard deviation of grayscale intensity)
    for quantitative evaluation in your report.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = round(float(np.mean(gray)), 2)
    contrast = round(float(np.std(gray)), 2)
    return {"brightness": brightness, "contrast": contrast}


def enhance_frame(
    frame,
    target_width=640,
    kernel_size=(5, 5),
    gamma=1.8,
    clip_limit=2.5,
    tile_grid_size=(8, 8),
    low_light_threshold=85.0,
    force_enhance=False
):
    """
    Full Phase 2 preprocessing pipeline:
    1. Resize frame
    2. Compute original metrics
    3. Denoise with Gaussian Blur
    4. If dark (or force_enhance=True), apply Gamma Correction + CLAHE
    """
    resized = resize_frame(frame, target_width)
    orig_metrics = compute_image_metrics(resized)

    denoised = apply_gaussian_blur(resized, kernel_size)

    is_low_light = orig_metrics["brightness"] < low_light_threshold
    if is_low_light or force_enhance:
        brightened = apply_gamma_correction(denoised, gamma=gamma)
        enhanced = apply_clahe(brightened, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    else:
        # Apply mild CLAHE even on well-lit frames to standardize contrast for YOLO
        enhanced = apply_clahe(denoised, clip_limit=1.5, tile_grid_size=tile_grid_size)

    enh_metrics = compute_image_metrics(enhanced)

    return resized, enhanced, {
        "is_low_light": is_low_light,
        "original": orig_metrics,
        "enhanced": enh_metrics
    }