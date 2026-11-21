"""AI-assisted clothing/shoulder reconstruction with a protected face region.

Only the lower clothing/shoulder polygon is editable by the image API. The
source face is composited back over the generated result, so the model cannot
replace facial details. Final output resizing/cropping can resample pixels;
this is not a bit-for-bit file-preservation guarantee.
"""
from __future__ import annotations

import io
import os
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageOps


DEFAULT_ENDPOINT = "https://api.openai.com/v1/images/edits"
DEFAULT_MODEL = "gpt-image-1"


def _make_edit_mask(size: tuple[int, int], face_box: tuple[int, int, int, int]) -> Image.Image:
    """Opaque = protected; transparent = editable per image-edit mask convention."""
    width, height = size
    fx, fy, fw, fh = face_box
    cx = fx + fw / 2.0
    chin_y = fy + fh * 0.96

    mask = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)
    # Start below the chin. Widen toward the shoulders so sloped or incomplete
    # garment edges can be rebuilt without allowing edits to the face/hair.
    polygon = [
        (int(cx - fw * 0.43), int(chin_y)),
        (int(cx + fw * 0.43), int(chin_y)),
        (int(cx + fw * 0.62), int(fy + fh * 1.18)),
        (int(cx + fw * 1.45), int(fy + fh * 1.52)),
        (width + 8, height + 8),
        (-8, height + 8),
        (int(cx - fw * 1.45), int(fy + fh * 1.52)),
        (int(cx - fw * 0.62), int(fy + fh * 1.18)),
    ]
    # Transparent fill marks the region the API may edit.
    draw.polygon(polygon, fill=(255, 255, 255, 0))
    return mask


def correct_clothing_shoulders(
    source: Image.Image,
    api_key: str,
    face_box: tuple[int, int, int, int] | None,
) -> Image.Image:
    """Rebuild clothing/shoulder area while restoring the untouched source face."""
    if face_box is None:
        raise ValueError(
            "AI clothing correction requires a detected face. No image was changed; "
            "use a clearer, front-facing source photo."
        )
    if not api_key or not api_key.strip():
        raise ValueError("An image API key is required for AI clothing correction.")

    endpoint = os.environ.get("EPIX_IMAGE_API_URL", DEFAULT_ENDPOINT).strip()
    model = os.environ.get("EPIX_IMAGE_MODEL", DEFAULT_MODEL).strip()
    if not endpoint or not model:
        raise ValueError("Image API endpoint and model must not be empty.")

    image = ImageOps.exif_transpose(source).convert("RGB")
    mask = _make_edit_mask(image.size, face_box)
    image_buffer, mask_buffer = io.BytesIO(), io.BytesIO()
    image.save(image_buffer, format="PNG")
    mask.save(mask_buffer, format="PNG")
    image_buffer.seek(0)
    mask_buffer.seek(0)

    prompt = (
        "Edit ONLY the transparent/masked lower clothing and shoulder region. "
        "The opaque region is protected and must remain unchanged. Preserve the "
        "person's original head, face, ears, hair, facial hair, expression, skin "
        "and neck exactly; do not redraw, retouch, beautify, rotate, reshape or "
        "reposition any facial or head pixels. Reconstruct or extend only the "
        "existing shirt and shoulder fabric where needed. Make the left and right "
        "shoulder silhouettes visually balanced and level, with equal apparent "
        "shoulder width, matching garment colour, fabric texture, lighting and "
        "perspective. Continue the real garment naturally to both lower corners; "
        "remove white triangular holes or transparent gaps inside the clothing. "
        "Do not invent a new garment design, logos, accessories, text or body "
        "details. Keep the original image composition and background unchanged "
        "outside the editable mask. Photorealistic, subtle, documentary-style edit."
    )

    try:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            data={
                "model": model,
                "prompt": prompt,
                "size": "auto",
                "output_format": "png",
            },
            files={
                "image": ("source.png", image_buffer.getvalue(), "image/png"),
                "mask": ("mask.png", mask_buffer.getvalue(), "image/png"),
            },
            timeout=(20, 240),
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Image API connection failed: {exc}") from exc

    if not response.ok:
        try:
            detail = response.json().get("error", {}).get("message", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(
            f"Image API returned HTTP {response.status_code}: {str(detail)[:1200]}"
        )

    try:
        import base64
        payload = response.json()
        generated = Image.open(io.BytesIO(base64.b64decode(
            payload["data"][0]["b64_json"], validate=True
        )))
        generated.load()
        generated = ImageOps.exif_transpose(generated).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            "The image API response did not contain a readable base64 image."
        ) from exc

    # Return to the source coordinate system. Restore the original face/head
    # region after generation; only pixels inside the clothing edit mask survive.
    if generated.size != image.size:
        generated = generated.resize(image.size, Image.Resampling.LANCZOS)

    editable = Image.new("L", image.size, 0)
    editable_draw = ImageDraw.Draw(editable)
    editable_polygon = [
        (int(face_box[0] + face_box[2] * 0.07), int(face_box[1] + face_box[3] * 0.98)),
        (int(face_box[0] + face_box[2] * 0.93), int(face_box[1] + face_box[3] * 0.98)),
        (int(face_box[0] + face_box[2] * 1.12), int(face_box[1] + face_box[3] * 1.35)),
        (image.width + 8, image.height + 8),
        (-8, image.height + 8),
        (int(face_box[0] - face_box[2] * 0.12), int(face_box[1] + face_box[3] * 1.35)),
    ]
    editable_draw.polygon(editable_polygon, fill=255)
    # Feather only the lower-clothing transition; the face remains fully
    # outside the editable polygon and is copied from the original image.
    editable = editable.filter(ImageFilter.GaussianBlur(radius=2.0))
    return Image.composite(generated, image, editable)
