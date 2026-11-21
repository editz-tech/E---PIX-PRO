"""Conservative face-roll and shoulder-line alignment for E-PIX PRO.

This module only performs geometric transforms. It does not synthesize, reshape,
beautify, or replace facial features. A true yaw change (turning a profile face
toward camera) is intentionally not attempted because unseen facial pixels
cannot be recovered while guaranteeing identity preservation.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


MAX_FACE_ROLL_DEGREES = 12.0
MAX_SHOULDER_CORRECTION_DEGREES = 8.0
SHOULDER_RESIDUAL_TOLERANCE = 1.25


def _rotate_expand(image: Image.Image, angle_degrees: float) -> Image.Image:
    """Rotate the complete portrait without cropping its corners."""
    rgba = np.asarray(image.convert("RGBA"))
    height, width = rgba.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = max(1, int(round(height * sine + width * cosine)))
    new_height = max(1, int(round(height * cosine + width * sine)))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    rotated = cv2.warpAffine(
        rgba,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(rotated, mode="RGBA")


def _shoulder_points(analyzer, image):
    """Return visible shoulder coordinates in image pixels, or None."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    result = analyzer._pose.process(rgb)
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks.landmark
    left = landmarks[11]
    right = landmarks[12]
    if min(left.visibility, right.visibility) < 0.55:
        return None
    p1 = (float(left.x * width), float(left.y * height))
    p2 = (float(right.x * width), float(right.y * height))
    if abs(p2[0] - p1[0]) < width * 0.08:
        return None
    return p1, p2


def _normalized_line_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return ((angle + 90.0) % 180.0) - 90.0


def _level_shoulders_below_face(image, analyzer, face_box):
    """Apply a smooth lower-body shear; keep the face region out of the warp."""
    points = _shoulder_points(analyzer, image)
    if points is None or face_box is None:
        return image, None

    p1, p2 = points
    angle = _normalized_line_angle(p1, p2)
    if abs(angle) < SHOULDER_RESIDUAL_TOLERANCE:
        return image, angle

    applied_angle = float(np.clip(
        angle,
        -MAX_SHOULDER_CORRECTION_DEGREES,
        MAX_SHOULDER_CORRECTION_DEGREES,
    ))
    slope = float(np.tan(np.radians(applied_angle)))

    width, height = image.size
    face_bottom = float(face_box[1] + face_box[3])
    shoulder_mid_y = (p1[1] + p2[1]) / 2.0
    vertical_gap = shoulder_mid_y - face_bottom
    if vertical_gap < max(8.0, face_box[3] * 0.035):
        return image, angle

    # Start below the chin/jaw boundary. The smooth transition reaches full
    # correction before the shoulder landmarks, so both are corrected equally.
    transition_start = face_bottom + vertical_gap * 0.06
    transition_end = face_bottom + vertical_gap * 0.48
    if transition_end <= transition_start:
        return image, angle

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    t = np.clip(
        (yy - transition_start) / (transition_end - transition_start),
        0.0,
        1.0,
    )
    blend = t * t * (3.0 - 2.0 * t)
    center_x = (p1[0] + p2[0]) / 2.0

    # Inverse mapping: the lower shoulder on one side is shifted upward and
    # the opposite side downward. The map is identity above the jaw boundary.
    map_x = xx
    map_y = yy + (slope * (xx - center_x) * blend).astype(np.float32)
    source = np.asarray(image.convert("RGBA"))
    corrected = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(corrected, mode="RGBA"), angle


def straighten_portrait(image: Image.Image, analyzer):
    """Level the eyes first, then independently level the shoulders.

    Returns (image, refreshed_face_box, applied_face_roll_degrees,
    final_shoulder_angle_degrees). The face is never generatively edited.
    """
    image = image.convert("RGBA")
    face_box, face_roll, shoulder_roll = analyzer.analyze(image)

    # Eye-line roll controls head orientation. If no face is detected, use the
    # shoulder line only as a conservative fallback; never rotate by guesswork.
    requested_roll = face_roll if face_box is not None else (
        shoulder_roll if shoulder_roll is not None else 0.0
    )
    applied_roll = 0.0
    if abs(requested_roll) >= 0.5:
        applied_roll = float(np.clip(
            requested_roll,
            -MAX_FACE_ROLL_DEGREES,
            MAX_FACE_ROLL_DEGREES,
        ))
        image = _rotate_expand(image, applied_roll)
        face_box, _, _ = analyzer.analyze(image)

    final_shoulder_angle = None
    # Re-detect after each correction. One residual pass handles small landmark
    # shifts from the first warp without accumulating aggressive distortion.
    for _ in range(2):
        image, measured_angle = _level_shoulders_below_face(
            image, analyzer, face_box
        )
        if measured_angle is None:
            final_shoulder_angle = None
            break
        final_shoulder_angle = measured_angle
        if abs(measured_angle) < SHOULDER_RESIDUAL_TOLERANCE:
            break
        face_box, _, _ = analyzer.analyze(image)

    face_box, _, _ = analyzer.analyze(image)
    return image, face_box, applied_roll, final_shoulder_angle
