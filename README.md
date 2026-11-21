# E-PIX PRO

Windows passport and ID photo studio with offline and optional full generative AI processing.

## Processing modes

### Offline mode
- Offline background segmentation using the bundled ONNX model
- White background compositing and 35 × 45 mm output
- Face-landmark framing and restrained geometric correction
- No generative face reconstruction

### Full Generative AI mode
- Sends the selected original photograph to an image-edit API.
- Regenerates the complete portrait to aim for a straight frontal view, level shoulders, balanced margins, complete shirt/shoulder silhouette, white background and 35:45 framing.
- The generated result is then face-detected and cropped to the application's output dimensions.
- This mode requires internet access and a valid API key. The key is entered in the app for the current run and is not saved by the application.
- The image is transmitted to the configured API provider for processing. Do not use this mode unless the person has consented and the provider's data handling terms are acceptable.

## API configuration

The UI uses the default image-edit endpoint and model configured in `ai_portrait.py`. They can be overridden with environment variables:

- `EPIX_IMAGE_API_URL`: full image-edit endpoint URL
- `EPIX_IMAGE_MODEL`: image model identifier supported by the API account

The current default model identifier in the source is `gpt-image-1`. Model access, account verification, rate limits, image-edit support and billing depend on the API account. The application does not include an API key; users must provide their own.

## Important identity and compliance limitations

Full generative mode is NOT a pixel-preserving photo correction. It synthesizes image content and may change facial geometry, eyes, nose, mouth, skin details, hair, expression or other identity cues even when the prompt asks it to retain the same person. Exact identity preservation is not guaranteed. The output must be compared with the source and approved by a human before use.

Generative edits can also invent or infer shirt/shoulder regions that were not visible in the source. This may make a visually balanced image, but it does not make the generated clothing or body anatomy documentary-accurate. Do not use generated results for identity verification or official documents where altered likeness is prohibited. Users must confirm the applicable authority's rules before submission.

The software should never describe a generated result as an exact or guaranteed match to the source person's face.

## Build

Install dependencies from `requirements.txt`. Package `app.py`, `ai_portrait.py`, and the `models/u2net.onnx` asset with the Windows application. Generative mode also requires internet access and an API key at runtime.
