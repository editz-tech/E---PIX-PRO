# E-PIX PRO

Windows passport and ID photo studio with an offline, identity-preserving portrait correction workflow.

## Identity-preserving portrait workflow

The desktop workflow is local/offline. It does not request an API key or transmit photographs to an image-generation service.

- Uses face landmarks to correct modest in-plane head roll.
- Detects both shoulder landmarks, corrects the measured shoulder-line slope (bounded to 12 degrees), and gradually centres the torso beneath the detected face. The body remap starts below the face box.
- Uses a wider face-relative crop (38% target face-box height) to show more upper body and shoulder area.
- Does not use generative AI to redraw or reconstruct facial features. When roll correction is applied, a rigid image rotation resamples pixels, but it does not synthesize or reshape facial features.
- Uses the bundled ONNX segmentation model for background removal and white-background compositing.

### Important limitations

Preserving the original face means the application cannot safely synthesize a new frontal face from a true yaw/pitch angle, change facial perspective, or reconstruct unseen facial details. It can correct modest 2D roll only. Shoulder correction is geometric, not generative. It depends on reliable shoulder landmarks and is bounded to avoid severe distortion; it cannot guarantee identical anatomical shoulder lengths for every pose or reconstruct a missing shoulder. Missing body/clothing outside the source frame cannot be recreated faithfully by this offline pipeline. Review every result before official use.

The former full-generative API integration is not part of the desktop processing workflow. No API key is requested or required.

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
