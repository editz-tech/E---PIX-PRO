from pathlib import Path
from urllib.request import urlopen

# U2NETP is the lightweight model chosen for faster offline CPU processing.
URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
target = Path("models/u2netp.onnx")
target.parent.mkdir(parents=True, exist_ok=True)

if not target.exists():
    print("Downloading lightweight U2NETP model for offline E-PIX PRO...")
    temporary = target.with_suffix(".onnx.download")
    try:
        with urlopen(URL, timeout=300) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if temporary.stat().st_size < 4_000_000:
            raise RuntimeError("Downloaded U2NETP model is unexpectedly small; refusing to package it.")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()

print(f"Lightweight U2NETP model ready: {target} ({target.stat().st_size:,} bytes)")
