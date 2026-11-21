"""OpenAI-compatible image-edit integration for E-PIX PRO full generative mode.

The whole portrait is regenerated. This is deliberately opt-in and must not be
described as identity-preserving: image generation can alter facial details.
"""
from __future__ import annotations

import base64
import io
import os

import requests
from PIL import Image, ImageOps


DEFAULT_ENDPOINT = "https://api.openai.com/v1/images/edits"
DEFAULT_MODEL = "gpt-image-1"


def generate_portrait(source: Image.Image, api_key: str) -> Image.Image:
    """Regenerate a formal ID portrait from the supplied source image.

    Environment overrides:
      EPIX_IMAGE_API_URL   full image-edit endpoint URL
      EPIX_IMAGE_MODEL     image model identifier supported by the API account

    The API key is supplied by the UI for the current run and is never written
    to disk or logged by this module.
    """
    if not api_key or not api_key.strip():
        raise ValueError("A valid image API key is required for generative mode.")

    endpoint = os.environ.get("EPIX_IMAGE_API_URL", DEFAULT_ENDPOINT).strip()
    model = os.environ.get("EPIX_IMAGE_MODEL", DEFAULT_MODEL).strip()
    if not endpoint or not model:
        raise ValueError("Image API endpoint and model must not be empty.")

    image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)

    prompt = (
        "Edit the supplied photograph into a professional, realistic, straight-on "
        "passport/ID portrait of the SAME PERSON. Create a genuinely frontal, "
        "upright head-and-shoulders pose: eyes level, nose centered, head vertical, "
        "chin naturally centered, both shoulders level and visually balanced, "
        "both shoulders visible to matching extents, equal left and right side "
        "margins, and a natural amount of space above the hair. Show the complete "
        "upper shoulders and upper chest. If the source shirt or shoulder edges "
        "are incomplete or tilted, plausibly reconstruct the missing lower clothing "
        "and shoulder silhouette using the same garment, fabric texture, colour, "
        "lighting and perspective. Do not leave white/transparent triangular gaps "
        "inside the clothing or along the lower corners. Use a clean, uniform pure "
        "white background, neutral even studio lighting, natural skin texture, "
        "realistic photographic detail, and restrained contrast. Keep the same "
        "hairstyle, facial hair, age appearance, clothing style and distinctive "
        "facial characteristics as closely as possible. Do not add accessories, "
        "text, logos, borders or props. Compose as a vertical 35:45 ID photograph. "
        "Important: this is a generative edit and may alter facial identity; do not "
        "claim pixel-identical or guaranteed identity preservation."
    )

    files = {"image": ("source.png", buffer.getvalue(), "image/png")}
    data = {
        "model": model,
        "prompt": prompt,
        "size": "1024x1536",
        "output_format": "png",
    }
    try:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            data=data,
            files=files,
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
        payload = response.json()
        encoded = payload["data"][0]["b64_json"]
        output_bytes = base64.b64decode(encoded, validate=True)
        generated = Image.open(io.BytesIO(output_bytes))
        generated.load()
        return ImageOps.exif_transpose(generated).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            "The image API response did not contain a readable base64 PNG. "
            "Check that the configured model supports image edits and base64 output."
        ) from exc
