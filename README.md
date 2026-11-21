# E-PIX PRO

Windows passport and ID photo studio using bundled offline AI processing.

## Processing mode

E-PIX PRO runs its portrait workflow locally using the bundled ONNX background-segmentation model and MediaPipe face/pose landmarks.

- No OpenAI API key or other API key is requested.
- No external image API is called.
- No subscription or internet connection is required for photo processing.
- Original input files are not overwritten.
- The workflow performs geometric alignment, background removal, white-background compositing and 35 × 45 mm output framing.

### Face preservation limitation

The offline workflow does not generatively redraw the face. It does rotate/align and crop the source image, so facial pixels can be resampled during geometric transforms and final resizing; this is not a guarantee of byte-for-byte pixel identity. It does not synthesize a new face. Review every output before official use.

The offline background-removal model segments the existing subject. It does not generate missing shoulders or invent new clothing. Shoulder correction is geometric and limited; if a source photo does not contain enough shoulder/body information, the software cannot reliably reconstruct it without generative image synthesis.

## Build

Install dependencies from `requirements.txt`. Package `app.py` and the `models/u2net.onnx` asset with the Windows application.

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
