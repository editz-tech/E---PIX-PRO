from PyInstaller.utils.hooks import collect_all
datas = [("models/u2netp.onnx", "models")]
binaries = []
hiddenimports = []
for package in ("onnxruntime", "PIL", "numpy"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
a = Analysis(["app.py"], pathex=[], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, hookspath=[], hooksconfig={},
             runtime_hooks=[], excludes=["rembg", "pymatting", "numba"], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="E-PIX PRO",
          debug=False, bootloader_ignore_signals=False, strip=False,
          upx=False, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="E-PIX PRO")
