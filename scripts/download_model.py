from pathlib import Path
from urllib.request import urlopen
URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
target = Path("models/u2netp.onnx")
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    print("Downloading segmentation model for offline installer...")
    with urlopen(URL, timeout=180) as response, target.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk: break
            output.write(chunk)
print(f"Model ready: {target} ({target.stat().st_size:,} bytes)")
