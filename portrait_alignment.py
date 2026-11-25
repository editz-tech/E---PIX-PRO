"""Offline face and shoulder alignment for E-PIX PRO.

Geometric-only transforms: facial features are never synthesized, reshaped or
locally warped. A true yaw change (turning a profile face toward camera) cannot
be recovered while guaranteeing identity preservation.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


MAX_FACE_ROLL_DEGREES = 12.0
MAX_SHOULDER_CORRECTION_DEGREES = 10.0
SHOULDER_RESIDUAL_TOLERANCE = 2.0


def _rotate_expand(image: Image.Image, angle_degrees: float) -> Image.Image:
    """Rotate the complete portrait without cropping its corners."""
    rgba = np.asarray(image.convert("RGBA"))
    height, width = rgba.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = max(1, int(round(height * sine + width * cosine)))
    new_height = max(1, int(round(height * cosine + width * sine)))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    rotated = cv2.warpAffine(
        rgba, matrix, (new_width, new_height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
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
    left, right = landmarks[11], landmarks[12]
    if min(left.visibility, right.visibility) < 0.55:
        return None
    p1 = (float(left.x * width), float(left.y * height))
    p2 = (float(right.x * width), float(right.y * height))
    if abs(p2[0] - p1[0]) < width * 0.08:
        return None
    return p1, p2


def _normalized_line_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return ((angle + 90.0) % 180.0) - 90.0


def _level_shoulders_below_face(image, analyzer, face_box, max_step=MAX_SHOULDER_CORRECTION_DEGREES):
    """Correct only clearly uneven shoulders; leave already-good shoulders intact.

    The supplied shoulder reference is used as a geometric guide: balanced,
    front-facing shoulder height and a centered upper torso. It is not pasted
    into the result, so each input keeps its own shirt, fabric texture and colour.
    Small natural shoulder slopes (under two degrees) are deliberately preserved.
    """
    points = _shoulder_points(analyzer, image)
    if points is None or face_box is None:
        return image, None

    p1, p2 = points
    measured = _normalized_line_angle(p1, p2)
    # Do not process every portrait indiscriminately. Preserve the original
    # shoulder/body pixels when the detected shoulder line is already within
    # the natural tolerance; only visibly uneven cases receive the warp.
    if abs(measured) < SHOULDER_RESIDUAL_TOLERANCE:
        return image, measured

    applied = float(np.clip(measured, -max_step, max_step))
    slope = float(np.tan(np.radians(applied)))
    width, height = image.size
    face_bottom = float(face_box[1] + face_box[3])
    shoulder_mid_y = (p1[1] + p2[1]) / 2.0
    gap = shoulder_mid_y - face_bottom
    if gap < max(8.0, face_box[3] * 0.035):
        return image, measured

    # A smooth transition begins below the jaw and reaches full correction
    # above the shoulder line, avoiding a visible seam across the neck.
    transition_start = face_bottom + gap * 0.04
    transition_end = face_bottom + gap * 0.52
    if transition_end <= transition_start:
        return image, measured

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    t = np.clip((yy - transition_start) / (transition_end - transition_start), 0.0, 1.0)
    blend = t * t * (3.0 - 2.0 * t)
    center_x = (p1[0] + p2[0]) / 2.0

    # Inverse-map both shoulders toward a level line. The map is identity
    # through the complete face region and changes smoothly below the jaw.
    map_x = xx
    map_y = yy + (slope * (xx - center_x) * blend).astype(np.float32)
    corrected = cv2.remap(
        np.asarray(image.convert("RGBA")), map_x, map_y,
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(corrected, mode="RGBA"), measured


def straighten_portrait(image: Image.Image, analyzer):
    """Correct eye-line roll, then independently level the shoulder line.

    Returns (image, refreshed_face_box, applied_face_roll_degrees,
    final_shoulder_angle_degrees). Face pixels are not generatively edited.
    """
    image = image.convert("RGBA")
    # FaceMesh is comparatively expensive. The face box is needed initially
    # and after whole-image rotation only; shoulder warps never touch the face.
    face_box, face_roll = analyzer.analyze_face(image)

    # Never infer face orientation from the shoulders when facial landmarks
    # are missing. This avoids rotating a face based on an unrelated body pose.
    applied_roll = 0.0
    if face_box is not None and abs(face_roll) >= 0.5:
        applied_roll = float(np.clip(face_roll, -MAX_FACE_ROLL_DEGREES, MAX_FACE_ROLL_DEGREES))
        image = _rotate_expand(image, applied_roll)
        face_box, _ = analyzer.analyze_face(image)

    final_shoulder_angle = None
    last_pass_warped = False
    # Limit shoulder correction to two measured passes, but do not rerun FaceMesh
    # after each lower-body warp: the face region and its crop bounds are intact.
    for _ in range(2):
        previous_image = image
        image, measured = _level_shoulders_below_face(
            image, analyzer, face_box, max_step=MAX_SHOULDER_CORRECTION_DEGREES
        )
        last_pass_warped = image is not previous_image
        if measured is None:
            final_shoulder_angle = None
            break
        final_shoulder_angle = measured
        if not last_pass_warped or abs(measured) < SHOULDER_RESIDUAL_TOLERANCE:
            break

    # Re-measure only if the fifth and final pass actually changed the image.
    if last_pass_warped:
        final_points = _shoulder_points(analyzer, image)
        final_shoulder_angle = (
            _normalized_line_angle(*final_points) if final_points is not None else None
        )
    return image, face_box, applied_roll, final_shoulder_angle
