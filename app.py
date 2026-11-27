from __future__ import annotations
import os, sys, traceback, time
from io import BytesIO
from pathlib import Path
import cv2
from portrait_alignment import straighten_portrait
from shadow_matte import build_subject_alpha
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, ImageFilter, ImageChops
from PySide6.QtCore import QObject, Signal, QThread, QTimer, Qt, QUrl, QSettings
from PySide6.QtGui import QPixmap, QDesktopServices, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QProgressBar, QCheckBox, QTextEdit, QGroupBox, QMessageBox, QFrame, QSizePolicy)

APP_NAME = "E-PIX PRO"
RATIO = 35 / 45
# Match the supplied demo image's pixel dimensions and preserve 35 × 45 mm print sizing.
OUTPUT_SIZE = (1103, 1426)
DPI = (800, 805)

def resource_path(name):
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / name

class U2NetSession:
    """Direct full U2Net inference without rembg, pymatting or numba."""
    def __init__(self, model_path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Avoid CPU oversubscription on common Windows desktops while keeping
        # ONNX inference parallel. One inter-op worker is enough for this graph.
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
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
        mask_image = build_subject_alpha(prediction, rgb)

        # Preserve transparent pixels introduced by expand=True rotation.
        # The RGB inference input cannot represent transparency, so explicitly
        # intersect the AI mask with the rotated source-image validity alpha.
        source_alpha = image.convert("RGBA").getchannel("A")
        mask_image = ImageChops.multiply(mask_image, source_alpha)

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

class FaceAnalyzer:
    """Face framing plus shoulder-line detection for level ID portraits."""
    def __init__(self):
        import mediapipe as mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.60
        )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=1,
            enable_segmentation=False, min_detection_confidence=0.55
        )

    def analyze_face(self, image):
        """Run only FaceMesh when face landmarks are needed."""
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        face_box, face_roll = None, 0.0
        face_result = self._mesh.process(rgb)
        if face_result.multi_face_landmarks:
            points = face_result.multi_face_landmarks[0].landmark
            xs = np.array([p.x * width for p in points], dtype=np.float32)
            ys = np.array([p.y * height for p in points], dtype=np.float32)
            x0, x1 = int(max(0, np.floor(xs.min()))), int(min(width, np.ceil(xs.max())))
            y0, y1 = int(max(0, np.floor(ys.min()))), int(min(height, np.ceil(ys.max())))
            # Estimate head roll from both eye centres, not just the outer corners.
            right_eye = ((points[33].x + points[133].x) / 2.0,
                         (points[33].y + points[133].y) / 2.0)
            left_eye = ((points[362].x + points[263].x) / 2.0,
                        (points[362].y + points[263].y) / 2.0)
            dx = (left_eye[0] - right_eye[0]) * width
            dy = (left_eye[1] - right_eye[1]) * height
            face_roll = float(np.degrees(np.arctan2(dy, dx)))
            if x1 > x0 and y1 > y0:
                face_box = (x0, y0, x1 - x0, y1 - y0)
        return face_box, face_roll

    def analyze_pose(self, image):
        """Run only Pose when shoulder landmarks are needed."""
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        shoulder_roll = None
        pose_result = self._pose.process(rgb)
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks.landmark
            left_shoulder, right_shoulder = lm[11], lm[12]
            if min(left_shoulder.visibility, right_shoulder.visibility) >= 0.45:
                dx = (right_shoulder.x - left_shoulder.x) * width
                dy = (right_shoulder.y - left_shoulder.y) * height
                angle = float(np.degrees(np.arctan2(dy, dx)))
                shoulder_roll = ((angle + 90.0) % 180.0) - 90.0
        return shoulder_roll

    def analyze(self, image):
        """Compatibility wrapper returning face and shoulder measurements."""
        face_box, face_roll = self.analyze_face(image)
        shoulder_roll = self.analyze_pose(image)
        return face_box, face_roll, shoulder_roll

    def close(self):
        self._mesh.close()
        self._pose.close()


def straighten_roll(image, analyzer):
    """Straighten the face first, then level only the lower shoulder/body region.

    The face is never locally warped. Expanded rotation uses replicated edge
    pixels instead of transparent fill, preventing white/transparent wedges
    from appearing in the shirt after background removal.
    """
    face_box, face_roll, shoulder_roll = analyzer.analyze(image)

    # The eye line is the authority for head alignment.
    if face_box is not None:
        roll = face_roll
    elif shoulder_roll is not None:
        roll = shoulder_roll
    else:
        roll = 0.0

    applied = max(-8.0, min(8.0, roll)) if abs(roll) >= 0.5 else 0.0

    if applied:
        # Expand the canvas while replicating boundary pixels. This is
        # important at the lower corners: shirt pixels are continued instead
        # of introducing transparent triangles that become white in the JPEG.
        arr = np.asarray(image.convert("RGBA"))
        h, w = arr.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, applied, 1.0)
        cos_a = abs(matrix[0, 0])
        sin_a = abs(matrix[0, 1])
        new_w = int(round((h * sin_a) + (w * cos_a)))
        new_h = int(round((h * cos_a) + (w * sin_a)))
        matrix[0, 2] += (new_w / 2.0) - center[0]
        matrix[1, 2] += (new_h / 2.0) - center[1]
        rotated = cv2.warpAffine(
            arr, matrix, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        image = Image.fromarray(rotated, mode="RGBA")
        face_box, _, shoulder_roll = analyzer.analyze(image)

    # Level the shoulder line with a gradual vertical shear confined to the
    # neck/body and shirt below the detected face. The face and hair region
    # remain untouched; no face pixels are reconstructed or reshaped.
    pose_result = analyzer._pose.process(np.asarray(image.convert("RGB")))
    if pose_result.pose_landmarks and face_box is not None:
        lm = pose_result.pose_landmarks.landmark
        ls, rs = lm[11], lm[12]
        if min(ls.visibility, rs.visibility) >= 0.55:
            h, w = image.height, image.width
            x1, y1 = ls.x * w, ls.y * h
            x2, y2 = rs.x * w, rs.y * h
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) > 1.0:
                shoulder_angle = float(np.degrees(np.arctan2(dy, dx)))
                # Do not apply extreme corrections to naturally sloped poses.
                limited_angle = max(-5.0, min(5.0, shoulder_angle))
                slope = float(np.tan(np.radians(limited_angle)))
                if abs(limited_angle) >= 0.8:
                    center_x = (x1 + x2) / 2.0
                    shoulder_y = (y1 + y2) / 2.0
                    face_bottom = float(face_box[1] + face_box[3])
                    start_y = min(float(h - 2), max(face_bottom + 2.0, 0.0))
                    end_y = max(start_y + 1.0, min(float(h - 1), shoulder_y))
                    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                    blend = np.clip((yy - start_y) / (end_y - start_y), 0.0, 1.0)
                    map_x = xx
                    map_y = yy + (slope * (xx - center_x) * blend).astype(np.float32)
                    src = np.asarray(image.convert("RGBA"))
                    leveled = cv2.remap(
                        src, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    image = Image.fromarray(leveled, mode="RGBA")

    # Refresh face bounds after the lower-body warp for accurate crop placement.
    face_box, _, _ = analyzer.analyze(image)
    return image, face_box, applied


def passport_crop(subject_rgba, face_box=None):
    """Build a face-scaled 35:45 ID portrait with safe, level edges.

    A fixed full-frame resize makes faces too small when the camera framing
    differs. When landmarks are available, derive the crop from face height,
    leave room above the forehead for hair and below the chin for shoulders,
    and centre horizontally on the face. Out-of-frame crop areas stay
    transparent and are composited as pure white, never as rotated wedges.
    """
    rgba = subject_rgba.convert("RGBA")
    src_w, src_h = rgba.size
    target_ratio = RATIO

    if face_box:
        fx, fy, fw, fh = face_box
        fh = max(1.0, float(fh))
        face_cx = float(fx) + float(fw) / 2.0

        # Match the approved reference framing consistently across batch jobs:
        # a larger, clearly readable head; visible hairline/headroom; balanced
        # side space around the head; and shoulders/upper chest retained below.
        # The prior 0.34 value made faces noticeably too small in the final frame.
        FACE_HEIGHT_TARGET = 0.45
        TOP_WHITE_MARGIN = 0.025
        crop_h = fh / FACE_HEIGHT_TARGET
        crop_w = crop_h * target_ratio
        left = face_cx - crop_w / 2.0

        # Detect thin flyaway hair as well as the dense hair mass. The old
        # row threshold ignored sparse crown strands and clipped the top margin.
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
        foreground_rows = np.count_nonzero(alpha > 48, axis=1)
        min_row_pixels = max(4, int(round(src_w * 0.0045)))
        visible_rows = np.flatnonzero(foreground_rows >= min_row_pixels)
        if visible_rows.size:
            hair_top = float(visible_rows[0])
            top = hair_top - (TOP_WHITE_MARGIN * crop_h)
        else:
            # Conservative fallback if the foreground mask is empty.
            top = float(fy) - (0.45 * fh)
        # Use the last substantial foreground row as the lower boundary.
        # This removes both out-of-bounds transparent padding and an existing
        # blank white strip at the bottom of the source cutout.
        content_bottom = float(visible_rows[-1] + 1) if visible_rows.size else float(src_h)
        top = min(top, content_bottom - crop_h)
        crop_box = (
            int(round(left)), int(round(top)),
            int(round(left + crop_w)), int(round(top + crop_h))
        )
        rgba = rgba.crop(crop_box)
    else:
        # Landmark fallback: use the AI foreground mask to remove excess
        # blank space above the hair before making the 35:45 crop. The old
        # centre-crop fallback preserved the camera's original head position;
        # this left some batch outputs with 15–25% empty space above the head.
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
        foreground_rows = np.count_nonzero(alpha > 48, axis=1)
        min_row_pixels = max(4, int(round(src_w * 0.0045)))
        visible_rows = np.flatnonzero(foreground_rows >= min_row_pixels)
        if visible_rows.size:
            hair_top = int(visible_rows[0])
            top = max(0, int(round(hair_top - (0.025 * src_h))))
            crop_h = max(1, src_h - top)
            crop_w = max(1, int(round(crop_h * target_ratio)))
            if crop_w <= src_w:
                # Keep the face centred and preserve a balanced amount of
                # space on both sides; the lower shoulders may naturally
                # reach the frame edges, as in the supplied reference.
                left = (src_w - crop_w) // 2
                rgba = rgba.crop((left, top, left + crop_w, src_h))
            else:
                crop_h = max(1, int(round(src_w / target_ratio)))
                crop_top = max(0, src_h - crop_h)
                rgba = rgba.crop((0, crop_top, src_w, crop_top + crop_h))
        else:
            # Empty/invalid mask: retain a safe centred ratio crop.
            source_ratio = src_w / max(1, src_h)
            if source_ratio > target_ratio:
                crop_w = max(1, int(round(src_h * target_ratio)))
                left = (src_w - crop_w) // 2
                rgba = rgba.crop((left, 0, left + crop_w, src_h))
            elif source_ratio < target_ratio:
                crop_h = max(1, int(round(src_w / target_ratio)))
                top = max(0, (src_h - crop_h) // 2)
                rgba = rgba.crop((0, top, src_w, top + crop_h))

    rgba = rgba.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", OUTPUT_SIZE, (255, 255, 255, 255))
    canvas.alpha_composite(rgba, (0, 0))
    return canvas.convert("RGB")

def process_photo(source, destination, session, face_analyzer):
    with Image.open(source) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGBA")

    original, face_box, roll, shoulder_angle = straighten_portrait(original, face_analyzer)
    cutout = session.remove_background(original)
    # Skip optional per-photo lighting/tone normalization for a faster, neutral pipeline.
    # Use the post-straightening face position to centre the final portrait.
    result = passport_crop(cutout, face_box)
    result = result.filter(ImageFilter.UnsharpMask(radius=0.9, percent=105, threshold=3))
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        result.save(destination, "JPEG", quality=96, dpi=DPI, optimize=False)
    elif suffix == ".png":
        # Lower compression spends less CPU; pixel data and output dimensions are unchanged.
        result.save(destination, "PNG", dpi=DPI, optimize=False, compress_level=3)
    elif suffix in (".tif", ".tiff"):
        result.save(destination, "TIFF", dpi=DPI, compression="tiff_lzw")
    elif suffix == ".bmp":
        result.save(destination, "BMP", dpi=DPI)
    elif suffix == ".webp":
        result.save(destination, "WEBP", quality=96, method=4)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")
    if face_box is None:
        return "face not detected — centered fallback; retake/review recommended"
    shoulder_status = "not detected — manual review required" if shoulder_angle is None else f"residual {shoulder_angle:.1f}°"
    review = " • REVIEW: shoulder line may still be uneven" if shoulder_angle is None or abs(shoulder_angle) > 2.0 else ""
    return f"face roll corrected {roll:.1f}°; shoulder line {shoulder_status}{review}"

class Worker(QObject):
    progress = Signal(int, int, str)
    done = Signal(int, int)
    failed = Signal(str)
    def __init__(self, files, output):
        super().__init__()
        self.files, self.output = files, output
    def run(self):
        try:
            model_dir = resource_path("models")
            os.environ["U2NET_HOME"] = str(model_dir)
            model = model_dir / "u2net.onnx"
            if not model.exists():
                raise RuntimeError("Offline AI model missing. Reinstall the application.")
            session = U2NetSession(model)
            face_analyzer = FaceAnalyzer()
            ok = errors = 0
            for i, src in enumerate(self.files, 1):
                try:
                    target = output_path_for(src, self.output)
                    framing = process_photo(src, target, session, face_analyzer); ok += 1
                    self.progress.emit(i, len(self.files), f"Saved ({framing}): {src.name} → {target.name}")
                except Exception as exc:
                    errors += 1; self.progress.emit(i, len(self.files), f"FAILED: {src.name}: {exc}")
            self.done.emit(ok, errors)
        except Exception: self.failed.emit(traceback.format_exc())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1600, 900)
        self.setMinimumSize(1180, 720)
        self.files = []
        self.settings = QSettings("EPIX", "E-PIX PRO")
        self.processing_started = None
        self.thread = None
        self.worker = None

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        page = QVBoxLayout(root)
        page.setContentsMargins(16, 10, 16, 12)
        page.setSpacing(10)

        # Brand header: render the supplied E-PIX logo as transparent vector artwork.
        header = QHBoxLayout()
        header.setSpacing(16)
        self.logo_label = QLabel()
        self.logo_label.setFixedSize(620, 190)
        self.logo_label.setStyleSheet("background:transparent;border:0;")
        logo_path = resource_path("assets/epix_logo.svg")
        logo_loaded = False
        if logo_path.exists():
            try:
                renderer = QSvgRenderer(str(logo_path))
                if not renderer.isValid():
                    raise ValueError("The packaged E-PIX logo SVG is invalid")
                logo_size = renderer.defaultSize().scaled(590, 176, Qt.KeepAspectRatio)
                pix = QPixmap(logo_size)
                pix.fill(Qt.transparent)
                painter = QPainter(pix)
                renderer.render(painter)
                painter.end()
                self.logo_label.setPixmap(pix)
                self.logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                logo_loaded = True
            except Exception as logo_error:
                print(f"Warning: E-PIX logo could not be loaded: {logo_error}")
        if not logo_loaded:
            self.logo_label.setText("E-PIX <span style='color:#1685ff'>PRO</span>")
            self.logo_label.setTextFormat(Qt.RichText)
            self.logo_label.setStyleSheet(
                "background:transparent;border:0;font-size:27px;font-weight:800;"
                "letter-spacing:2px;color:#ffffff;"
            )
            self.logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.addWidget(self.logo_label)
        header.addStretch(1)

        self.home_button = QPushButton("⌂  Home")
        self.home_button.setObjectName("navActive")
        self.home_button.clicked.connect(lambda: self.status.setText("Home • Ready to process"))
        settings_nav = QPushButton("⚙  Settings")
        settings_nav.clicked.connect(self.choose_output)
        about_nav = QPushButton("ⓘ  About")
        about_nav.clicked.connect(self.show_about)
        for b in (self.home_button, settings_nav, about_nav):
            b.setMinimumHeight(42)
            header.addWidget(b)
        page.addLayout(header)

        hero = QHBoxLayout()
        hero.addStretch(1)
        self.selection_summary = QLabel("0 PHOTOS SELECTED")
        self.selection_summary.setStyleSheet("color:#f2f2f2;font-size:9pt;font-weight:700;letter-spacing:1px;")
        hero.addWidget(self.selection_summary)
        page.addLayout(hero)

        content = QHBoxLayout()
        content.setSpacing(16)

        # Left card: live source-photo preview.
        preview_card = QFrame()
        preview_card.setObjectName("panel")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        preview_layout.setSpacing(12)
        preview_head = QLabel("PHOTO PREVIEW")
        preview_head.setObjectName("panelTitle")
        preview_layout.addWidget(preview_head)
        self.preview = QLabel("Choose photos to preview\n\n35 × 45 mm ID format")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(220, 250)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setObjectName("photoPreview")
        preview_layout.addWidget(self.preview, 1)
        content.addWidget(preview_card, 5)

        # Centre card: quick actions and output destination.
        actions_card = QFrame()
        actions_card.setObjectName("panel")
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(16, 16, 16, 16)
        actions_layout.setSpacing(14)
        actions_title = QLabel("QUICK ACTIONS")
        actions_title.setObjectName("panelTitle")
        actions_layout.addWidget(actions_title)
        actions_grid = QGridLayout()
        actions_grid.setSpacing(10)

        add_tile = QPushButton("▧  Add Photos")
        add_tile.clicked.connect(self.choose_multiple)
        folder_tile = QPushButton("▣  Choose Folder")
        folder_tile.clicked.connect(self.choose_folder)
        output_tile = QPushButton("▤  Output Folder")
        output_tile.clicked.connect(self.choose_output)
        results_tile = QPushButton("◉  View Result")
        results_tile.clicked.connect(self.open_output_folder)
        for tile in (add_tile, folder_tile, output_tile, results_tile):
            tile.setMinimumHeight(76)
            tile.setCursor(Qt.PointingHandCursor)
            tile.setStyleSheet(tile.styleSheet() + "QPushButton {font-size:11pt;font-weight:700;padding:8px;}")
        add_tile.setObjectName("primary")
        actions_grid.addWidget(add_tile, 0, 0)
        actions_grid.addWidget(folder_tile, 0, 1)
        actions_grid.addWidget(output_tile, 1, 0)
        actions_grid.addWidget(results_tile, 1, 1)
        actions_layout.addLayout(actions_grid)

        # Keep the selected destination internally. A folder becomes the
        # permanent launch default only when the user enables the checkbox.
        fallback_output = str(Path.home() / "Pictures" / "E-Pix Output")
        saved_output = str(self.settings.value("default_output", fallback_output))
        # Migrate older installs: previously selected folders were always saved.
        saved_default_enabled = self.settings.value(
            "default_output_enabled", self.settings.contains("default_output"),
            type=bool
        )
        self.output = QLineEdit(saved_output if saved_default_enabled else fallback_output)
        self.output.hide()
        self.default_output_checkbox = QCheckBox("Set as default")
        self.default_output_checkbox.setChecked(saved_default_enabled)
        self.default_output_checkbox.setToolTip(
            "Remember this output folder and automatically use it whenever E-PIX PRO opens."
        )
        self.default_output_checkbox.toggled.connect(self.set_default_output_enabled)
        actions_grid.addWidget(self.default_output_checkbox, 2, 0, alignment=Qt.AlignLeft)

        self.preserve = QCheckBox("Preserve original filenames")
        self.preserve.setChecked(True)
        self.preserve.setEnabled(False)
        actions_grid.addWidget(self.preserve, 2, 1, alignment=Qt.AlignLeft)
        clear_button = QPushButton("Clear Selection")
        clear_button.clicked.connect(self.clear_selection)
        clear_button.setMinimumHeight(52)
        actions_layout.addWidget(clear_button)
        content.addWidget(actions_card, 5)

        # Right card: batch progress and elapsed timer.
        progress_card = QFrame()
        progress_card.setObjectName("panel")
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(16, 16, 16, 16)
        progress_layout.setSpacing(13)
        progress_title = QLabel("PROCESSING")
        progress_title.setObjectName("panelTitle")
        progress_layout.addWidget(progress_title)
        self.progress_percent = QLabel("0%")
        self.progress_percent.setAlignment(Qt.AlignCenter)
        self.progress_percent.setStyleSheet("font-size:32px;font-weight:800;color:#ffffff;")
        progress_layout.addWidget(self.progress_percent)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(13)
        progress_layout.addWidget(self.progress)
        self.status = QLabel("Ready to process")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#a8bad0;font-weight:600;")
        progress_layout.addWidget(self.status)
        progress_layout.addSpacing(12)
        timer_caption = QLabel("ELAPSED TIME")
        timer_caption.setObjectName("panelTitle")
        progress_layout.addWidget(timer_caption)
        self.elapsed_label = QLabel("00:00:00")
        self.elapsed_label.setStyleSheet("font-size:25px;font-weight:800;color:#ffffff;")
        progress_layout.addWidget(self.elapsed_label)
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(250)
        self.elapsed_timer.timeout.connect(self.update_elapsed_time)
        self.batch_label = QLabel("Total: 0 photos")
        self.batch_label.setStyleSheet("color:#8ea6c2;")
        progress_layout.addWidget(self.batch_label)
        progress_layout.addStretch(1)
        self.start = QPushButton("▶  START PROCESSING")
        self.start.setObjectName("primary")
        self.start.setMinimumHeight(52)
        self.start.clicked.connect(self.start_processing)
        progress_layout.addWidget(self.start)
        content.addWidget(progress_card, 3)
        page.addLayout(content, 1)

        # Activity log remains available below the dashboard.
        log_group = QGroupBox("ACTIVITY / RESULTS")
        log_layout = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(84)
        self.log.setPlaceholderText("Processing messages and saved-photo results will appear here.")
        log_layout.addWidget(self.log)
        page.addWidget(log_group)


        self.setStyleSheet("""
            QMainWindow, QWidget#root { background:#000000; color:#f5f5f5; font-family:"Segoe UI"; }
            QWidget { color:#f5f5f5; font-family:"Segoe UI"; font-size:10pt; }
            QFrame#panel {
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #101010,stop:0.45 #050505,stop:1 #0b0b0b);
                border:1px solid #555555; border-radius:18px;
            }
            QLabel#panelTitle { color:#f4f4f4; font-size:10pt; font-weight:800; letter-spacing:0.5px; }
            QLabel#photoPreview { background:#020202; border:1px dashed #b7b7b7; border-radius:12px; color:#eeeeee; font-size:12pt; }
            QPushButton {
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #242424,stop:0.5 #080808,stop:1 #171717);
                color:#f7f7f7; border:1px solid #777777; border-radius:13px;
                padding:10px 12px; min-height:20px; font-weight:650;
            }
            QPushButton:hover { background:#292929; border-color:#c8c8c8; }
            QPushButton:pressed { background:#050505; }
            QPushButton:disabled { color:#777777; background:#101010; border-color:#333333; }
            QPushButton#primary, QPushButton#navActive {
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #292929,stop:0.5 #080808,stop:1 #191919);
                border:1px solid #a0a0a0; color:#ffffff; font-weight:800;
            }
            QPushButton#primary:hover, QPushButton#navActive:hover { background:#303030; }
            QLineEdit, QTextEdit {
                background:#050505; color:#f5f5f5; border:1px solid #777777;
                border-radius:10px; padding:9px; selection-background-color:#555555;
            }
            QLineEdit:focus, QTextEdit:focus { border:1px solid #bdbdbd; }
            QProgressBar { background:#383838; border:0; border-radius:7px; }
            QProgressBar::chunk { background:#d8d8d8; border-radius:7px; }
            QCheckBox { spacing:7px; color:#f0f0f0; font-size:9pt; }
            QCheckBox::indicator { width:15px; height:15px; border:1px solid #b5b5b5; border-radius:3px; background:#080808; }
            QCheckBox::indicator:checked { background:#eeeeee; border:1px solid #ffffff; }
            QGroupBox {
                background:#080808; border:1px solid #555555; border-radius:15px;
                margin-top:12px; padding:12px; color:#f2f2f2; font-weight:800;
            }
            QGroupBox::title { subcontrol-origin:margin; left:14px; padding:0 6px; }
        """)

    def set_files(self, files):
        self.files = list(dict.fromkeys(
            Path(f) for f in files
            if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        ))
        count = len(self.files)
        self.selection_summary.setText(f"{count} PHOTOS SELECTED")
        self.batch_label.setText(f"Total: {count} photos")
        if self.files:
            pix = QPixmap(str(self.files[0]))
            if not pix.isNull():
                self.preview.setPixmap(pix.scaled(
                    self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                self.preview.setText("")
            else:
                self.preview.setPixmap(QPixmap())
                self.preview.setText("Preview unavailable")
        else:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Choose photos to preview\n\n35 × 45 mm ID format")

    def clear_selection(self):
        if self.start.isEnabled():
            self.set_files([])
            self.log.clear()
            self.progress.setValue(0)
            self.progress_percent.setText("0%")
            self.status.setText("Ready to process")
            self.elapsed_label.setText("00:00:00")

    def choose_single(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Choose photo", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
        )
        if f:
            self.set_files([f])

    def choose_multiple(self):
        fs, _ = QFileDialog.getOpenFileNames(
            self, "Choose multiple photos", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
        )
        if fs:
            self.set_files(fs)

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose photo folder")
        if d:
            self.set_files([p for p in Path(d).iterdir() if p.is_file()])

    def choose_output(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output.text())
        if d:
            self.output.setText(d)
            if self.default_output_checkbox.isChecked():
                self.settings.setValue("default_output", d)
                self.settings.setValue("default_output_enabled", True)

    def set_default_output_enabled(self, enabled):
        """Persist the chosen destination when the default checkbox is enabled."""
        self.settings.setValue("default_output_enabled", bool(enabled))
        if enabled:
            self.settings.setValue("default_output", self.output.text().strip())
        self.settings.sync()

    def open_output_folder(self):
        folder = Path(self.output.text()).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def show_about(self):
        QMessageBox.about(
            self, "About E-PIX PRO",
            "E-PIX PRO\nProfessional offline passport and ID photo studio.\n35 × 45 mm output."
        )

    def start_processing(self):
        if not self.files:
            QMessageBox.warning(self, "No photos", "Choose one or more photos first.")
            return
        self.start.setEnabled(False)
        self.progress.setMaximum(max(1, len(self.files)))
        self.progress.setValue(0)
        self.progress_percent.setText("0%")
        self.log.clear()
        self.processing_started = time.perf_counter()
        self.elapsed_label.setText("00:00:00")
        self.elapsed_timer.start()
        self.status.setText("Starting offline processing…")
        self.thread = QThread()
        self.worker = Worker(self.files, Path(self.output.text()).expanduser())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.done.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def update_elapsed_time(self):
        if self.processing_started is None:
            return
        seconds = int(time.perf_counter() - self.processing_started)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.elapsed_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def on_progress(self, current, total, message):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(current)
        percent = int((current / max(1, total)) * 100)
        self.progress_percent.setText(f"{percent}%")
        self.status.setText(f"Processing {current} of {total}")
        self.log.append(message)

    def on_done(self, ok, errors):
        self.elapsed_timer.stop()
        self.update_elapsed_time()
        self.start.setEnabled(True)
        self.status.setText(f"Finished • {ok} successful • {errors} failed")
        self.log.append(f"Finished: {ok} successful, {errors} failed.")
        self.batch_label.setText(f"Completed: {ok} photos • Failed: {errors}")

    def on_failed(self, message):
        self.elapsed_timer.stop()
        self.update_elapsed_time()
        self.start.setEnabled(True)
        self.status.setText("Processing could not start.")
        self.log.append(message)
        QMessageBox.critical(
            self, "Processing error",
            "Offline processing engine could not start. See activity log for details."
        )

if __name__ == "__main__":
    app = QApplication(sys.argv); window = MainWindow(); window.show(); sys.exit(app.exec())
