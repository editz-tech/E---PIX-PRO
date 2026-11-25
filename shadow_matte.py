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


def _guided_alpha_refine(alpha, rgb):
    """Align the soft AI alpha to visible image edges without hard thresholding."""
    image = np.asarray(rgb, dtype=np.uint8)
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    source = np.asarray(alpha, dtype=np.float32)

    # Small-radius guided filter: preserve fine hair transitions while
    # preventing the low-resolution AI mask from bleeding across strong edges.
    radius = 2
    kernel = (radius * 2 + 1, radius * 2 + 1)
    mean_i = cv2.boxFilter(guide, cv2.CV_32F, kernel, normalize=True)
    mean_p = cv2.boxFilter(source, cv2.CV_32F, kernel, normalize=True)
    corr_i = cv2.boxFilter(guide * guide, cv2.CV_32F, kernel, normalize=True)
    corr_ip = cv2.boxFilter(guide * source, cv2.CV_32F, kernel, normalize=True)
    variance = np.maximum(corr_i - mean_i * mean_i, 0.0)
    covariance = corr_ip - mean_i * mean_p
    a = covariance / (variance + 0.0015)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, kernel, normalize=True)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, kernel, normalize=True)
    refined = mean_a * guide + mean_b

    # Keep most of the model's soft matte; use the guided result only to
    # correct edge placement, not to turn hair into a hard cutout.
    return np.clip(source * 0.68 + refined * 0.32, 0.0, 1.0)


def _decontaminate_edge_rgb(rgb, alpha):
    """Remove old-background colour spill from partially transparent edge pixels.

    The nearest high-confidence foreground colour is used as a conservative
    estimate of the true edge colour. Only a narrow band is adjusted; fully
    opaque skin, hair and clothing pixels are never recoloured.
    """
    image = np.asarray(rgb, dtype=np.uint8)
    sure_fg = (alpha >= 0.92).astype(np.uint8)
    if not np.any(sure_fg):
        return image

    # OpenCV labels each zero pixel (the confident foreground) so each
    # partially transparent edge pixel can borrow colour from its nearest
    # reliable interior pixel.
    distance_input = (1 - sure_fg).astype(np.uint8)
    distance, labels = cv2.distanceTransformWithLabels(
        distance_input, cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_PIXEL
    )
    ys, xs = np.where(sure_fg > 0)
    label_values = labels[ys, xs]
    max_label = int(labels.max())
    lookup = np.zeros((max_label + 1, 3), dtype=np.uint8)
    valid = (label_values > 0) & (label_values <= max_label)
    lookup[label_values[valid]] = image[ys[valid], xs[valid]]

    edge = (
        (alpha >= 0.10) & (alpha < 0.92) &
        (distance <= 7.0) & (labels > 0) & (labels <= max_label)
    )
    if np.any(edge):
        nearest = lookup[labels[edge]].astype(np.float32)
        original = image[edge].astype(np.float32)
        # Stronger cleanup at the faint fringe; retain some source detail in
        # semitransparent hair and fine fabric fibres.
        strength = np.clip((0.92 - alpha[edge]) / 0.72, 0.0, 1.0)[:, None]
        cleaned = original * (1.0 - 0.78 * strength) + nearest * (0.78 * strength)
        result = image.copy()
        result[edge] = np.clip(cleaned, 0, 255).astype(np.uint8)
        return result
    return image


def build_subject_alpha(raw_prediction, rgb):
    """Create an edge-aware subject matte and clean RGB fringe in-place."""
    raw = np.clip(raw_prediction, 0.0, 1.0)
    raw_full = cv2.resize(raw, rgb.size, interpolation=cv2.INTER_CUBIC)
    raw_full = np.clip(raw_full, 0.0, 1.0)
    mask = np.clip((raw_full - 0.16) / (0.78 - 0.16), 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)

    rgb_array = np.asarray(rgb, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    saturation = hsv[:, :, 1]
    mean = cv2.boxFilter(gray, cv2.CV_32F, (9, 9), normalize=True)
    mean_sq = cv2.boxFilter(gray * gray, cv2.CV_32F, (9, 9), normalize=True)
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    sure_fg = (raw_full >= 0.72).astype(np.uint8)
    if np.any(sure_fg):
        outside = (1 - sure_fg).astype(np.uint8)
        distance = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
        band_width = max(8.0, min(rgb.size) * 0.022)
        shadow_band = (
            (sure_fg == 0) &
            (distance <= band_width) &
            (raw_full < 0.76) &
            (saturation < 60) &
            (gray >= 35.0) & (gray <= 242.0) &
            (local_std < 88.0)
        )
        mask[shadow_band] = 0.0

    mask = _guided_alpha_refine(mask, rgb)

    # Final shadow-halo rejection AFTER guided filtering. Guided filters can
    # otherwise pull low-opacity dark pixels back into the matte. Suppress only
    # low-chroma, low-texture pixels just outside the confident subject core;
    # never suppress pixels inside the core, so dark hair/skin/clothing remain.
    sure_core = (raw_full >= 0.82).astype(np.uint8)
    if np.any(sure_core):
        outside_core = (1 - sure_core).astype(np.uint8)
        dist_to_core = cv2.distanceTransform(outside_core, cv2.DIST_L2, 3)
        shadow_band_width = max(10.0, min(rgb.size) * 0.018)
        # Compare candidate pixels to the local confident-subject luminance.
        core_luma = np.where(sure_core > 0, gray, 0.0).astype(np.float32)
        core_count = cv2.boxFilter(sure_core.astype(np.float32), cv2.CV_32F,
                                   (31, 31), normalize=False)
        core_sum = cv2.boxFilter(core_luma, cv2.CV_32F, (31, 31),
                                 normalize=False)
        local_core_luma = core_sum / np.maximum(core_count, 1.0)
        candidate_shadow = (
            (sure_core == 0) &
            (dist_to_core > 1.0) &
            (dist_to_core <= shadow_band_width) &
            (saturation < 72) &
            (local_std < 72.0) &
            (gray < np.maximum(105.0, local_core_luma - 28.0)) &
            (raw_full < 0.88)
        )
        # Do not leave a translucent dark fringe behind ears, hair or shoulders.
        mask[candidate_shadow] = 0.0

    cleaned_rgb = _decontaminate_edge_rgb(rgb_array, mask)
    # rgb is a private RGB copy created by U2NetSession; update it before the
    # RGBA foreground is assembled so new backgrounds do not inherit the
    # original white/grey/blue edge fringe.
    rgb.paste(Image.fromarray(cleaned_rgb, mode="RGB"))

    mask_image = Image.fromarray(
        np.uint8(np.clip(mask, 0.0, 1.0) * 255), mode="L"
    )
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=0.18))
    return mask_image


def remove_baked_ear_halo(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    erode_px: int = 2,
    feather_sigma: float = 0.7,
    bg_whiten_min_val: int = 165,
    bg_max_chroma: int = 28,
    bg_color=(255, 255, 255),
):
    """Remove baked-in grey/dark fringe using the real foreground alpha mask.

    rgb is the cropped/resized RGB image. mask is its matching subject alpha
    (0 background, 255 foreground). This runs immediately before compositing.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    if mask is None:
        max_c = np.max(rgb, axis=2).astype(np.int16)
        min_c = np.min(rgb, axis=2).astype(np.int16)
        chroma = max_c - min_c
        is_bg_like = (max_c >= bg_whiten_min_val) & (chroma <= bg_max_chroma)
        fg = (~is_bg_like).astype(np.uint8) * 255
        fg = cv2.morphologyEx(
            fg, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        )
    else:
        mask = np.asarray(mask)
        if mask.shape[:2] != (h, w):
            raise ValueError("Halo-removal mask dimensions must match the RGB image.")
        if mask.dtype == np.uint8:
            fg = (mask > 127).astype(np.uint8) * 255
        else:
            scale = 255.0 if float(np.nanmax(mask)) <= 1.0 else 1.0
            fg = (mask.astype(np.float32) * scale > 127).astype(np.uint8) * 255

    if erode_px > 0:
        k_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
        )
        fg_clean = cv2.erode(fg, k_erode, iterations=1)
    else:
        fg_clean = fg.copy()

    alpha = fg_clean.astype(np.float32) / 255.0
    if feather_sigma > 0:
        ksize = max(3, int(round(feather_sigma * 4)) | 1)
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), feather_sigma)
        alpha[alpha < 0.08] = 0.0

    ring = cv2.dilate(
        (fg_clean > 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
    )
    edge_band = ((ring > 0) & ((fg_clean == 0) | (alpha < 0.95))).astype(np.uint8)
    rgb_clean = rgb.copy()
    if np.any(edge_band):
        rgb_clean = cv2.inpaint(
            rgb_clean, edge_band, inpaintRadius=2, flags=cv2.INPAINT_TELEA
        )

    bg = np.empty_like(rgb_clean, dtype=np.float32)
    bg[:] = np.asarray(bg_color, dtype=np.float32)
    a3 = alpha[..., None]
    out = a3 * rgb_clean.astype(np.float32) + (1.0 - a3) * bg
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)

    max_c = np.max(out_u8, axis=2).astype(np.int16)
    min_c = np.min(out_u8, axis=2).astype(np.int16)
    near_gray_bg = (
        (alpha < 0.15) &
        (max_c >= bg_whiten_min_val) &
        ((max_c - min_c) <= bg_max_chroma)
    )
    out_u8[near_gray_bg] = np.asarray(bg_color, dtype=np.uint8)
    return out_u8
