# E-PIX PRO

Windows passport and ID photo studio with offline processing and optional face-protected AI clothing/shoulder correction.

## Processing modes

### Offline mode
- Offline background segmentation using the bundled ONNX model
- White background compositing and 35 × 45 mm output
- Face-landmark framing and restrained geometric correction
- No generative face reconstruction

### AI-assisted clothing and shoulder correction (face-protected)
- Sends the selected original photograph and a protected-region edit mask to the image-edit API.
- Only lower clothing and shoulder regions are eligible for generative reconstruction; the head/face region is protected by the mask and restored locally from the source after the API response.
- The mode can rebuild missing shirt corners and balance the visible shoulder silhouette. It does not rotate or reshape the face.
- Requires internet access and a valid API key. The key is entered for the current run and is not saved by the application.
- The image is transmitted to the configured API provider. Obtain consent and review the provider's data-handling terms before use.

## API configuration

The UI uses the default image-edit endpoint and model configured in `ai_clothing.py`. They can be overridden with environment variables:

- `EPIX_IMAGE_API_URL`: full image-edit endpoint URL
- `EPIX_IMAGE_MODEL`: image model identifier supported by the API account

The current default model identifier in the source is `gpt-image-1`. Model access, account verification, rate limits, image-edit support and billing depend on the API account. The application does not include an API key; users must provide their own.

## Important identity and compliance limitations

Full generative mode is NOT a pixel-preserving photo correction. It synthesizes image content and may change facial geometry, eyes, nose, mouth, skin details, hair, expression or other identity cues even when the prompt asks it to retain the same person. Exact identity preservation is not guaranteed. The output must be compared with the source and approved by a human before use.

Generative edits can also invent or infer shirt/shoulder regions that were not visible in the source. This may make a visually balanced image, but it does not make the generated clothing or body anatomy documentary-accurate. Do not use generated results for identity verification or official documents where altered likeness is prohibited. Users must confirm the applicable authority's rules before submission.

The software should never describe a generated result as an exact or guaranteed match to the source person's face.

## Build

Install dependencies from `requirements.txt`. Package `app.py`, `ai_clothing.py`, and the `models/u2net.onnx` asset with the Windows application. AI clothing/shoulder mode also requires internet access and an API key at runtime.


## Portable Windows distribution and download safety

The GitHub Actions portable workflow performs these checks before publishing an artifact:

- Confirms that the application executable and bundled ONNX model exist.
- Creates the portable ZIP and extracts it again to verify that the expected files are present.
- Runs Microsoft Defender signature updates and custom scans against the build folder and ZIP. Publication stops if Defender reports a new detection or the scan tools are unavailable.
- Publishes a SHA-256 checksum file and a build manifest containing the source commit, executable hash, ZIP hash, Authenticode status and scan result.

### Important: this build is unsigned

This project currently has no trusted Windows code-signing certificate. The executable is therefore expected to report as unsigned. Microsoft Defender reporting no detections is useful build hygiene, but it is not a malware-free guarantee and does not establish a publisher reputation. Chrome Safe Browsing and Windows SmartScreen make their own reputation and threat decisions; a checksum does not override those decisions. The workflow cannot guarantee that a browser will allow the download.

Do not disable Chrome Safe Browsing, Microsoft Defender, SmartScreen or other security controls to install this application. If a download is blocked, stop and review the warning rather than trying to bypass it. A trusted code-signing certificate and a consistent release history are the appropriate longer-term route to publisher reputation, but signing alone does not guarantee that every security product will immediately trust a new release.

### Verify a downloaded portable ZIP

Keep `E-PIX-PRO-Portable.zip` and `SHA256SUMS.txt` together. In PowerShell, run:

```powershell
(Get-FileHash .\E-PIX-PRO-Portable.zip -Algorithm SHA256).Hash
Get-Content .\SHA256SUMS.txt
```

The hash printed by the first command must exactly match the 64-character hash in `SHA256SUMS.txt`. This confirms that the ZIP matches the checksum published in the same GitHub Actions artifact; it does not prove that the application is safe or signed. For stronger provenance, compare the build manifest's source commit with the repository commit used for the workflow run.
