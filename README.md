# E-PIX PRO

Offline Windows passport and ID photo studio.

## Planned first release
- Single-photo and multi-photo/folder workflows
- Offline background segmentation using a bundled ONNX model
- White background compositing and 35 × 45 mm output at 300 DPI
- Preserve source files and original filenames; save to a separate output folder
- Collision-safe output handling, progress and error reporting

## Important limitations
This tool does not synthesize a frontal pose, change gaze, close a mouth, or modify clothing. It preserves the subject from the source image and provides a review warning when a source photo may not meet ID-photo requirements. Users must verify the applicable authority's current specifications before submission.

## Build
The Windows GitHub Actions workflow will package the app and its segmentation model so processing works offline after installation.
