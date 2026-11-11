from __future__ import annotations
import os, sys, traceback
from pathlib import Path
from PIL import Image, ImageOps
from rembg import new_session, remove
from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QProgressBar, QCheckBox, QTextEdit, QGroupBox, QMessageBox)

APP_NAME = "E-PIX PRO"
RATIO = 35 / 45
DPI = 300

def resource_path(name):
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / name

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
    cutout = remove(original, session=session).convert("RGBA")
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
            session = new_session("u2netp", model_path=str(model))
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
        self.setStyleSheet("QMainWindow,QWidget{background:#101820;color:#e8eef5;} QGroupBox{border:1px solid #34495e;border-radius:8px;margin-top:8px;padding:10px;} QPushButton{padding:9px;background:#263746;border:1px solid #456078;border-radius:5px;} QPushButton:hover{background:#34516a;} QLineEdit,QTextEdit{background:#17232e;border:1px solid #34495e;padding:7px;}")
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
