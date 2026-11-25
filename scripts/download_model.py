from pathlib import Path
from urllib.request import urlopen

# Full U2Net is larger than U2NetP but retains more fine foreground detail,
# especially around hair, at the cost of additional download and RAM usage.
URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
target = Path("models/u2net.onnx")
target.parent.mkdir(parents=True, exist_ok=True)

if not target.exists():
    print("Downloading full U2Net model for offline E-PIX PRO...")
    temporary = target.with_suffix(".onnx.download")
    try:
        with urlopen(URL, timeout=300) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if temporary.stat().st_size < 50_000_000:
            raise RuntimeError("Downloaded U2Net model is unexpectedly small; refusing to package it.")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()

print(f"Full U2Net model ready: {target} ({target.stat().st_size:,} bytes)")
