# E-PIX PRO

Windows passport and ID photo studio with fully local processing.

## Offline processing

- U2Net ONNX performs local foreground/background segmentation.
- MediaPipe Face Mesh estimates face bounds and eye-line roll.
- MediaPipe Pose estimates shoulder landmarks.
- OpenCV applies bounded head rotation and a gradual lower-body shoulder-line correction.
- Pillow creates a white background and exports 35 × 45 mm framing at the configured output resolution.
- No image is sent to an external service. No API key, API billing or runtime internet connection is required.

## Face and clothing limitations

The software does not use a cloud image-generation API and does not synthesize new clothing. The local pose/segmentation models can align visible shoulders, preserve the visible garment, and remove the background. They cannot reliably invent missing shirt fabric or reconstruct an entirely absent shoulder. Use a source photograph that includes both shoulders and sufficient shirt area when complete clothing is required.

Face landmarks are used to estimate roll. A geometric rotation and final crop/resize can resample facial pixels; therefore, the result is not bit-for-bit identical to the source. The application does not ask a generative model to redraw the face and does not apply global brightness, contrast, colour or sharpening filters in the offline processing path. Review every result and confirm the applicable issuing authority's photo rules.

## Runtime and build

Install dependencies from `requirements.txt`. The Windows package includes `models/u2net.onnx`. The model is downloaded during the build workflow and bundled with the application; end users do not need internet access during processing.

## Portable Windows distribution and download safety

The GitHub Actions portable workflow validates the executable and bundled ONNX model, extracts the ZIP to verify its contents, scans the build with Microsoft Defender, and publishes SHA-256 checksums and a build manifest.

### Important: this build is unsigned

This project currently has no trusted Windows code-signing certificate. The executable is therefore expected to report as unsigned. A clean Defender scan is useful build hygiene, but it is not a malware-free guarantee and does not establish publisher reputation. Chrome Safe Browsing and Windows SmartScreen make their own threat and reputation decisions. Do not disable security controls to install this application.

### Verify a downloaded portable ZIP

Keep `E-PIX-PRO-Portable.zip` and `SHA256SUMS.txt` together. In PowerShell:

```powershell
(Get-FileHash .\E-PIX-PRO-Portable.zip -Algorithm SHA256).Hash
Get-Content .\SHA256SUMS.txt
```

The hashes must match. This confirms integrity against the supplied checksum, not that the application is safe or signed.
