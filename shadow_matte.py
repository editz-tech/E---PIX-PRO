"""Edge-aware offline foreground matte refinement for E-PIX PRO.

The U2Net alpha is refined against source-image edges and RGB fringe from the
original background is decontaminated before compositing onto a new colour.
"""
import cv2
import numpy as np
from PIL import Image, ImageFilter


def fix_ear_halo(
    rgb: np.ndarray,
    alpha: np.ndarray,
    low_alpha_cutoff: float = 0.35,
    feather_sigma: float = 0.6,
    erode_px: int = 0,
):
    """Remove dark/grey matte fringe while preserving alpha for final compositing.

    This is the alpha-preserving adaptation of the supplied ear-halo cleanup.
    The final white/background composite remains in passport_crop so the UI's
    selectable background colours continue to work. Erosion is opt-in because
    shrinking the matte by default can clip fine hair, ears and clothing.
    """
    if alpha.dtype == np.uint8:
        a = alpha.astype(np.float32) / 255.0
    else:
        a = np.clip(alpha.astype(np.float32).copy(), 0.0, 1.0)

    # Remove only faint, low-opacity fringe; retain the main soft silhouette.
    a[a < low_alpha_cutoff] = 0.0

    # Optional mask contraction for cases with a clearly visible residual rim.
    # Keep disabled by default to avoid cutting natural ear/hair contours.
    if erode_px > 0:
        binary = (a > 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
        )
        eroded_fg = cv2.erode(binary, kernel, iterations=1)
        a = np.where(eroded_fg > 0, a, 0.0)

    if feather_sigma > 0:
        ksize = max(3, int(round(feather_sigma * 4)) | 1)
        a = cv2.GaussianBlur(a, (ksize, ksize), feather_sigma)
        a[a < 0.05] = 0.0

    # Decontaminate RGB only in the narrow anti-aliased boundary band.
    edge_band = ((a > 0.05) & (a < 0.98)).astype(np.uint8)
    cleaned = np.asarray(rgb, dtype=np.uint8).copy()
    if np.any(edge_band):
        cleaned = cv2.inpaint(cleaned, edge_band, inpaintRadius=2,
                              flags=cv2.INPAINT_TELEA)

    return cleaned, np.uint8(np.clip(a, 0.0, 1.0) * 255.0)


def build_subject_alpha(raw_prediction, rgb):
    """Fast edge matte: refine at bounded resolution, then upscale once.

    U2NetP predicts a 320x320 mask. Refining at at most 640px on the long
    edge avoids repeatedly filtering 3K/4K source images while retaining
    enough resolution for clean ID-photo hair, ear and shoulder contours.
    """
    original_size = rgb.size
    scale = min(1.0, 640.0 / max(original_size))
    work_size = (
        max(1, int(round(original_size[0] * scale))),
        max(1, int(round(original_size[1] * scale))),
    )
    work_rgb = rgb.resize(work_size, Image.Resampling.BILINEAR) if work_size != original_size else rgb
    raw = np.clip(raw_prediction, 0.0, 1.0)
    raw_work = cv2.resize(raw, work_size, interpolation=cv2.INTER_CUBIC)
    raw_work = np.clip(raw_work, 0.0, 1.0)
    mask = np.clip((raw_work - 0.16) / (0.78 - 0.16), 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)

    rgb_array = np.asarray(work_rgb, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    saturation = hsv[:, :, 1]
    mean = cv2.boxFilter(gray, cv2.CV_32F, (9, 9), normalize=True)
    mean_sq = cv2.boxFilter(gray * gray, cv2.CV_32F, (9, 9), normalize=True)
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    sure_fg = (raw_work >= 0.72).astype(np.uint8)
    if np.any(sure_fg):
        distance = cv2.distanceTransform(1 - sure_fg, cv2.DIST_L2, 3)
        band_width = max(4.0, min(work_size) * 0.022)
        shadow_band = (
            (sure_fg == 0) & (distance <= band_width) &
            (raw_work < 0.76) & (saturation < 60) &
            (gray >= 35.0) & (gray <= 242.0) & (local_std < 88.0)
        )
        mask[shadow_band] = 0.0

    # Suppress dark, low-texture shadow halos immediately outside the subject.
    sure_core = (raw_work >= 0.82).astype(np.uint8)
    if np.any(sure_core):
        dist_to_core = cv2.distanceTransform(1 - sure_core, cv2.DIST_L2, 3)
        shadow_width = max(5.0, min(work_size) * 0.018)
        core_luma = np.where(sure_core > 0, gray, 0.0).astype(np.float32)
        core_count = cv2.boxFilter(sure_core.astype(np.float32), cv2.CV_32F,
                                   (21, 21), normalize=False)
        core_sum = cv2.boxFilter(core_luma, cv2.CV_32F, (21, 21),
                                 normalize=False)
        local_core_luma = core_sum / np.maximum(core_count, 1.0)
        candidate_shadow = (
            (sure_core == 0) & (dist_to_core > 1.0) &
            (dist_to_core <= shadow_width) & (saturation < 72) &
            (local_std < 72.0) &
            (gray < np.maximum(105.0, local_core_luma - 28.0)) &
            (raw_work < 0.88)
        )
        mask[candidate_shadow] = 0.0

    mask_image = Image.fromarray(
        np.uint8(np.clip(mask, 0.0, 1.0) * 255), mode="L"
    )
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=0.22))
    if work_size != original_size:
        mask_image = mask_image.resize(original_size, Image.Resampling.LANCZOS)
    return mask_image
