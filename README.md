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


## Windows installer distribution

The `.github/workflows/build-windows.yml` workflow creates an x64 Inno Setup installer. It installs E-PIX PRO under Program Files, creates a Start Menu shortcut, offers an optional desktop shortcut, and registers a normal uninstaller. The installer bundles the offline ONNX model with the application.

Before publishing, the workflow validates the executable and model, compiles the installer, checks that the installer output exists and is a plausible size, scans the application folder and installer with Microsoft Defender, and publishes a SHA-256 checksum plus a build manifest. The GitHub Actions artifact is named `E-PIX-PRO-Windows-Installer` and contains `E-PIX-PRO-Setup.exe`, `SHA256SUMS.txt`, and `BUILD-MANIFEST.json`.

The installer is not digitally signed because this project does not have a trusted Windows code-signing certificate. Chrome Safe Browsing and Windows SmartScreen may still warn or block the download. A clean Defender scan and matching checksum do not guarantee safety or publisher reputation. Do not disable security controls to install it. Download only from the repository's GitHub Actions artifact for a successful workflow run, and verify the SHA-256 value before use.

The portable ZIP remains available separately from the `Build E-PIX PRO portable version` workflow.
