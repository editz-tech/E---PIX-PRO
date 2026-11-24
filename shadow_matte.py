"""Selective offline foreground matte cleanup for neutral cast-shadow halos."""
import cv2
import numpy as np
from PIL import Image, ImageFilter


def build_subject_alpha(raw_prediction, rgb):
    """Create a soft foreground alpha and selectively suppress smooth grey cast shadows."""
    # Build a soft subject matte, then remove only likely cast-shadow
    # pixels in a narrow band OUTSIDE the high-confidence subject core.
    # This is intentionally not a global threshold/erosion: dark hair,
    # saturated skin, ears, neck and clothing remain protected.
    raw = np.clip(raw_prediction, 0.0, 1.0)
    raw_full = cv2.resize(raw, rgb.size, interpolation=cv2.INTER_CUBIC)
    raw_full = np.clip(raw_full, 0.0, 1.0)
    mask = np.clip((raw_full - 0.16) / (0.78 - 0.16), 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)

    rgb_array = np.asarray(rgb, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    saturation = hsv[:, :, 1]
    # Local texture distinguishes smooth studio cast shadows from hair.
    mean = cv2.boxFilter(gray, cv2.CV_32F, (9, 9), normalize=True)
    mean_sq = cv2.boxFilter(gray * gray, cv2.CV_32F, (9, 9), normalize=True)
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    sure_fg = (raw_full >= 0.72).astype(np.uint8)
    if np.any(sure_fg):
        outside = (1 - sure_fg).astype(np.uint8)
        distance = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
        band_width = max(8.0, min(rgb.size) * 0.022)
        # Grey/near-black cast shadows are low-chroma and smoother than
        # hair strands. Include edge-adjacent pixels in the variance limit;
        # a narrow box window otherwise mistakes the shadow/white transition
        # for texture and leaves the halo untouched. Dark textured hair is
        # protected by the low-confidence and local-variance constraints.
        shadow_band = (
            (sure_fg == 0) &
            (distance <= band_width) &
            (raw_full < 0.76) &
            (saturation < 60) &
            (gray >= 35.0) & (gray <= 242.0) &
            # A cast shadow next to a bright white background naturally has
            # a higher local variance at its outer edge. The previous 48 limit
            # rejected exactly those pixels, so the halo remained. This wider
            # limit still rejects highly textured hair/fabric; near-black
            # pixels and saturated skin are protected by the other tests.
            (local_std < 88.0)
        )
        mask[shadow_band] = 0.0

    mask_image = Image.fromarray(np.uint8(np.clip(mask, 0.0, 1.0) * 255), mode="L")
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=0.28))
    return mask_image

