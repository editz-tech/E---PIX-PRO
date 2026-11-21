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

The API edit mask protects the head/face region, and the application composites the original source back over that protected region after generation. This prevents AI facial redraw. Final crop/resize can still resample pixels, so the exported face is not bit-for-bit identical to the source file. AI clothing correction intentionally skips geometric face straightening; a genuinely turned or tilted face remains as photographed because straightening it would alter facial pixels.

AI may reconstruct missing shirt/shoulder fabric or infer unseen body areas. Review all generated clothing and shoulder anatomy carefully. Do not use generated clothing/body regions where documentary accuracy is required, and confirm the applicable authority's rules before submitting official identity documents.

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
