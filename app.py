from __future__ import annotations
import os, sys, traceback
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, ImageFilter
from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QProgressBar, QCheckBox, QTextEdit, QGroupBox, QMessageBox)

APP_NAME = "E-PIX PRO"
RATIO = 35 / 45
DPI = 300

def resource_path(name):
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / name

class U2NetSession:
    """Direct U2NETP inference without rembg, pymatting or numba."""
    def __init__(self, model_path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def remove_background(self, image):
        rgb = image.convert("RGB")
        resized = rgb.resize((320, 320), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = ((pixels - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)
        prediction = self.session.run(None, {self.input_name: tensor})[0][0, 0]
        low, high = float(prediction.min()), float(prediction.max())
        if high - low < 1e-8:
            raise RuntimeError("AI model returned an invalid foreground mask.")
        mask = np.clip((prediction - low) / (high - low), 0.0, 1.0)
        mask_image = Image.fromarray(np.uint8(mask * 255), mode="L")
        mask_image = mask_image.resize(rgb.size, Image.Resampling.BICUBIC)
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=0.45))
        rgba = rgb.convert("RGBA")
        rgba.putalpha(mask_image)
        return rgba


def output_path_for(source, output_dir):
    # Keep the exact original filename. Never silently append suffixes or overwrite.
    target = output_dir / source.name
    if target.resolve() == source.resolve():
        raise FileExistsError("Output folder is the same as the source location; choose a separate output folder.")
    if target.exists():
        raise FileExistsError(f"An output file named '{source.name}' already exists. Move it or choose another output folder.")
    return target

def process_photo(source, destination, session):
    with Image.open(source) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGBA")
    cutout = session.remove_background(original)
    background = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
    background.alpha_composite(cutout)
    result = background.convert("RGB")
    w, h = result.size
    if w / h > RATIO:
        nw = int(h * RATIO); left = (w - nw) // 2
        result = result.crop((left, 0, left + nw, h))
    else:
        nh = int(w / RATIO); top = (h - nh) // 2
        result = result.crop((0, top, w, top + nh))
    # Standard 35 × 45 mm at 300 DPI (approximately 413 × 531 pixels).
    result = result.resize((413, 531), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        result.save(destination, "JPEG", quality=95, dpi=(DPI, DPI), optimize=True)
    elif suffix == ".png":
        result.save(destination, "PNG", dpi=(DPI, DPI), optimize=True)
    elif suffix in (".tif", ".tiff"):
        result.save(destination, "TIFF", dpi=(DPI, DPI), compression="tiff_lzw")
    elif suffix == ".bmp":
        result.save(destination, "BMP", dpi=(DPI, DPI))
    elif suffix == ".webp":
        result.save(destination, "WEBP", quality=95, method=6)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")

class Worker(QObject):
    progress = Signal(int, int, str)
    done = Signal(int, int)
    failed = Signal(str)
    def __init__(self, files, output):
        super().__init__(); self.files, self.output = files, output
    def run(self):
        try:
            model_dir = resource_path("models")
            os.environ["U2NET_HOME"] = str(model_dir)
            model = model_dir / "u2netp.onnx"
            if not model.exists(): raise RuntimeError("Offline AI model missing. Reinstall the application.")
            session = U2NetSession(model)
            ok = errors = 0
            for i, src in enumerate(self.files, 1):
                try:
                    target = output_path_for(src, self.output)
                    process_photo(src, target, session); ok += 1
                    self.progress.emit(i, len(self.files), f"Saved: {src.name} → {target.name}")
                except Exception as exc:
                    errors += 1; self.progress.emit(i, len(self.files), f"FAILED: {src.name}: {exc}")
            self.done.emit(ok, errors)
        except Exception: self.failed.emit(traceback.format_exc())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle(APP_NAME); self.resize(900, 650)
        self.files = []
        root = QWidget(); self.setCentralWidget(root); layout = QVBoxLayout(root)
        title = QLabel("E-PIX PRO"); title.setStyleSheet("font-size:28px;font-weight:700;color:#58a6ff;")
        layout.addWidget(title)
        layout.addWidget(QLabel("OFFLINE PASSPORT & ID PHOTO STUDIO  •  35 × 45 mm  •  300 DPI"))
        group = QGroupBox("1. Select photos"); row = QHBoxLayout(group)
        self.count = QLabel("No photos selected")
        for label, handler in [("Choose Photo", self.choose_single), ("Choose Multiple Photos", self.choose_multiple), ("Choose Folder", self.choose_folder)]:
            b = QPushButton(label); b.clicked.connect(handler); row.addWidget(b)
        row.addWidget(self.count); layout.addWidget(group)
        outbox = QGroupBox("2. Output folder"); outrow = QHBoxLayout(outbox)
        self.output = QLineEdit(str(Path.home() / "Pictures" / "E-Pix Output"))
        browse = QPushButton("Browse"); browse.clicked.connect(self.choose_output)
        outrow.addWidget(self.output); outrow.addWidget(browse); layout.addWidget(outbox)
        self.preserve = QCheckBox("Preserve original filename (required)"); self.preserve.setChecked(True); self.preserve.setEnabled(False); layout.addWidget(self.preserve)
        self.start = QPushButton("START PROCESSING"); self.start.setMinimumHeight(48)
        self.start.setStyleSheet("background:#1769aa;color:white;font-weight:700;border-radius:7px;")
        self.start.clicked.connect(self.start_processing); layout.addWidget(self.start)
        self.progress = QProgressBar(); layout.addWidget(self.progress)
        self.status = QLabel("Ready • Original photos are never modified."); layout.addWidget(self.status)
        self.log = QTextEdit(); self.log.setReadOnly(True); layout.addWidget(self.log)
        note = QLabel("AI removes the background and adds white. It does not change pose, gaze, expression, clothing or facial features. Review each photo before official use.")
        note.setWordWrap(True); note.setStyleSheet("color:#e3b341;"); layout.addWidget(note)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0b1220;
                color: #e7edf7;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            QLabel { color: #dce6f3; }
            QGroupBox {
                background: #111c2d;
                border: 1px solid #263a55;
                border-radius: 12px;
                margin-top: 14px;
                padding: 14px 12px 12px 12px;
                font-size: 10pt;
                font-weight: 600;
                color: #9fc5f5;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QPushButton {
                background: #1b2b40;
                color: #edf5ff;
                border: 1px solid #36516f;
                border-radius: 8px;
                padding: 10px 14px;
                min-height: 18px;
                font-weight: 600;
            }
            QPushButton:hover { background: #263f5c; border-color: #5b91c8; }
            QPushButton:pressed { background: #142338; }
            QPushButton:disabled { color: #718096; background: #172235; border-color: #263246; }
            QLineEdit, QTextEdit {
                background: #0d1726;
                color: #edf4ff;
                border: 1px solid #2c405a;
                border-radius: 7px;
                padding: 9px;
                selection-background-color: #2878bd;
            }
            QLineEdit:focus, QTextEdit:focus { border: 1px solid #4b9ce2; }
            QProgressBar {
                background: #172438;
                border: 1px solid #2d425d;
                border-radius: 7px;
                min-height: 14px;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk { background: #2587ce; border-radius: 6px; }
            QCheckBox { spacing: 8px; color: #b9c9dc; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid #52708f;
                border-radius: 4px;
                background: #0d1726;
            }
            QCheckBox::indicator:checked { background: #2587ce; border-color: #2587ce; }
        """)
    def set_files(self, files):
        self.files = list(dict.fromkeys(Path(f) for f in files if Path(f).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}))
        self.count.setText(f"{len(self.files)} photo(s) selected")
    def choose_single(self):
        f,_ = QFileDialog.getOpenFileName(self,"Choose photo","","Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if f: self.set_files([f])
    def choose_multiple(self):
        fs,_ = QFileDialog.getOpenFileNames(self,"Choose multiple photos","","Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if fs: self.set_files(fs)
    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self,"Choose photo folder")
        if d: self.set_files([p for p in Path(d).iterdir() if p.is_file()])
    def choose_output(self):
        d = QFileDialog.getExistingDirectory(self,"Choose output folder",self.output.text())
        if d: self.output.setText(d)
    def start_processing(self):
        if not self.files: QMessageBox.warning(self,"No photos","Choose one or more photos first."); return
        self.start.setEnabled(False); self.progress.setValue(0); self.log.clear()
        self.thread = QThread(); self.worker = Worker(self.files, Path(self.output.text()).expanduser())
        self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress); self.worker.done.connect(self.on_done); self.worker.failed.connect(self.on_failed)
        self.worker.done.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater); self.thread.start()
    def on_progress(self, current, total, message):
        self.progress.setMaximum(total); self.progress.setValue(current); self.status.setText(f"Processing {current} of {total}"); self.log.append(message)
    def on_done(self, ok, errors):
        self.start.setEnabled(True); self.status.setText(f"Finished • {ok} successful • {errors} failed"); self.log.append(f"Finished: {ok} successful, {errors} failed.")
    def on_failed(self, message):
        self.start.setEnabled(True); self.status.setText("Processing could not start."); self.log.append(message)
        QMessageBox.critical(self,"Processing error","Offline processing engine could not start. See log for details.")

if __name__ == "__main__":
    app = QApplication(sys.argv); window = MainWindow(); window.show(); sys.exit(app.exec())
