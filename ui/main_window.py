import os
from PySide6.QtCore import Qt, QThread, Signal, QRectF
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox,
    QProgressBar, QTextEdit, QGroupBox, QMessageBox, QSplitter
)

from core.inference import InferenceEngine
from core.exporter import AnnotationExporter
from core.i18n import LANGUAGE_OPTIONS, tr


class WorkerThread(QThread):
    """
    Thread separada para rodar a inferência sem travar a interface gráfica.
    """
    progress = Signal(int, int, str, dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, engine, input_dir, conf):
        super().__init__()
        self.engine = engine
        self.input_dir = input_dir
        self.conf = conf

    def run(self):
        try:
            valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
            image_files = [
                os.path.join(self.input_dir, f)
                for f in sorted(os.listdir(self.input_dir))
                if f.lower().endswith(valid_extensions)
            ]

            results = {}
            total = len(image_files)

            for i, img_path in enumerate(image_files):
                res = self.engine.process_image(img_path, conf_threshold=self.conf)
                if res:
                    results[os.path.basename(img_path)] = res
                self.progress.emit(i + 1, total, img_path, res or {})

            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lang = "en"
        self.inference_engine = None
        self.results_data = {}
        self.worker = None

        self.resize(1020, 620)
        self.init_ui()
        self.retranslate_ui()
        self.init_engine_async()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Coluna Esquerda: Controles e Logs ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Grupo 1: Seleção de Pastas
        self.dir_group = QGroupBox()
        dir_layout = QVBoxLayout()

        self.input_path_input = QLineEdit()
        self.btn_browse_input = QPushButton()
        self.btn_browse_input.clicked.connect(self.select_input_dir)

        input_h = QHBoxLayout()
        input_h.addWidget(self.input_path_input)
        input_h.addWidget(self.btn_browse_input)
        dir_layout.addLayout(input_h)

        self.output_path_input = QLineEdit()
        self.btn_browse_output = QPushButton()
        self.btn_browse_output.clicked.connect(self.select_output_dir)

        output_h = QHBoxLayout()
        output_h.addWidget(self.output_path_input)
        output_h.addWidget(self.btn_browse_output)
        dir_layout.addLayout(output_h)

        self.dir_group.setLayout(dir_layout)
        left_layout.addWidget(self.dir_group)

        # Grupo 2: Configurações
        self.settings_group = QGroupBox()
        settings_layout = QVBoxLayout()

        model_h = QHBoxLayout()
        self.model_label = QLabel()
        self.model_path_input = QLineEdit("yolov8n.pt")
        self.btn_browse_model = QPushButton()
        self.btn_browse_model.clicked.connect(self.select_model_file)
        model_h.addWidget(self.model_label)
        model_h.addWidget(self.model_path_input)
        model_h.addWidget(self.btn_browse_model)
        settings_layout.addLayout(model_h)

        params_h = QHBoxLayout()
        self.format_label = QLabel()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["YOLO TXT", "LabelMe JSON", "COCO JSON"])
        params_h.addWidget(self.format_label)
        params_h.addWidget(self.format_combo)

        self.conf_label = QLabel()
        self.conf_input = QLineEdit("0.25")
        self.conf_input.setMaximumWidth(65)
        params_h.addWidget(self.conf_label)
        params_h.addWidget(self.conf_input)
        settings_layout.addLayout(params_h)

        lang_h = QHBoxLayout()
        self.language_label = QLabel()
        self.lang_combo = QComboBox()
        for display_name, _ in LANGUAGE_OPTIONS:
            self.lang_combo.addItem(display_name)
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        lang_h.addWidget(self.language_label)
        lang_h.addWidget(self.lang_combo)
        settings_layout.addLayout(lang_h)

        self.settings_group.setLayout(settings_layout)
        left_layout.addWidget(self.settings_group)

        # Execução e Progresso
        self.btn_run = QPushButton()
        self.btn_run.setStyleSheet(
            "background-color: #2b5b84; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_run.clicked.connect(self.start_processing)
        left_layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        # Console de Logs
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        left_layout.addWidget(self.console)

        # --- Coluna Direita: Visualizador de Rótulos ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.preview_group = QGroupBox()
        preview_layout = QVBoxLayout()

        nav_h = QHBoxLayout()
        self.preview_image_label = QLabel()
        self.image_selector = QComboBox()
        self.image_selector.currentTextChanged.connect(self.on_select_preview_image)
        nav_h.addWidget(self.preview_image_label)
        nav_h.addWidget(self.image_selector, stretch=1)
        preview_layout.addLayout(nav_h)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(460, 420)
        self.preview_label.setStyleSheet(
            "background-color: #181a1f; color: #b0b7c3; border: 1px solid #323741; border-radius: 4px;"
        )
        preview_layout.addWidget(self.preview_label, stretch=1)

        self.preview_group.setLayout(preview_layout)
        right_layout.addWidget(self.preview_group)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([460, 560])

    def t(self, key: str, **kwargs) -> str:
        return tr(key, self.current_lang, **kwargs)

    def on_language_changed(self, index: int):
        if 0 <= index < len(LANGUAGE_OPTIONS):
            _, lang_code = LANGUAGE_OPTIONS[index]
            self.current_lang = lang_code
            self.retranslate_ui()

    def retranslate_ui(self):
        self.setWindowTitle(self.t("window_title"))
        self.dir_group.setTitle(self.t("dir_group"))
        self.input_path_input.setPlaceholderText(self.t("input_placeholder"))
        self.output_path_input.setPlaceholderText(self.t("output_placeholder"))
        self.btn_browse_input.setText(self.t("btn_browse_input"))
        self.btn_browse_output.setText(self.t("btn_browse_output"))

        self.settings_group.setTitle(self.t("settings_group"))
        self.model_label.setText(self.t("model_label"))
        self.model_path_input.setPlaceholderText(self.t("model_placeholder"))
        self.btn_browse_model.setText(self.t("btn_browse_model"))
        self.language_label.setText(self.t("language_label"))
        self.format_label.setText(self.t("format_label"))
        self.conf_label.setText(self.t("conf_label"))

        self.btn_run.setText(self.t("btn_run"))
        self.console.setPlaceholderText(self.t("console_placeholder"))

        self.preview_group.setTitle(self.t("preview_group"))
        self.preview_image_label.setText(self.t("preview_image_label"))
        if not self.results_data:
            self.preview_label.setText(self.t("preview_empty"))

    def log(self, text):
        self.console.append(text)

    def init_engine_async(self):
        self.log(self.t("log_ready"))

    def select_input_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, self.t("dialog_select_input"))
        if dir_name:
            self.input_path_input.setText(dir_name)

    def select_output_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, self.t("dialog_select_output"))
        if dir_name:
            self.output_path_input.setText(dir_name)

    def select_model_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            self.t("dialog_select_model"),
            "",
            self.t("dialog_model_filter")
        )
        if file_name:
            self.model_path_input.setText(file_name)

    def _color_for_class(self, class_id):
        hue = (class_id * 67) % 360
        return QColor.fromHsl(hue, 220, 140)

    def render_annotated_pixmap(self, image_path, annotations):
        qimg = QImage(image_path)
        if qimg.isNull():
            return None

        qimg = qimg.convertToFormat(QImage.Format_RGB888)
        painter = QPainter(qimg)
        painter.setRenderHint(QPainter.Antialiasing)

        font = QFont("Sans Serif", max(10, int(qimg.height() * 0.022)))
        font.setBold(True)
        painter.setFont(font)

        pen_width = max(2, int(min(qimg.width(), qimg.height()) * 0.004))

        for ann in annotations:
            x1, y1, x2, y2 = ann["bbox"]
            cls_id = ann.get("class_id", 0)
            label = f"{ann['class_name']} {ann['confidence']:.2f}"
            color = self._color_for_class(cls_id)

            pen = QPen(color, pen_width)
            painter.setPen(pen)
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(label) + 10
            text_h = metrics.height() + 4
            label_y = y1 - text_h if (y1 - text_h) >= 0 else y1

            painter.fillRect(QRectF(x1, label_y, text_w, text_h), color)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRectF(x1 + 5, label_y, text_w - 5, text_h), Qt.AlignVCenter | Qt.AlignLeft, label)

        painter.end()

        pixmap = QPixmap.fromImage(qimg)
        return pixmap.scaled(
            max(100, self.preview_label.width() - 12),
            max(100, self.preview_label.height() - 12),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

    def update_preview(self, img_name, res_data):
        if not res_data or "image_path" not in res_data:
            return
        pixmap = self.render_annotated_pixmap(
            res_data["image_path"],
            res_data.get("annotations", [])
        )
        if pixmap:
            self.preview_label.setPixmap(pixmap)

    def on_select_preview_image(self, img_name):
        if img_name and img_name in self.results_data:
            self.update_preview(img_name, self.results_data[img_name])

    def start_processing(self):
        input_dir = self.input_path_input.text().strip()
        output_dir = self.output_path_input.text().strip()

        if not input_dir or not os.path.exists(input_dir):
            QMessageBox.critical(self, self.t("dialog_error_title"), self.t("err_invalid_input"))
            return

        if not output_dir:
            QMessageBox.critical(self, self.t("dialog_error_title"), self.t("err_invalid_output"))
            return

        try:
            conf = float(self.conf_input.text().replace(",", "."))
        except ValueError:
            conf = 0.25

        model_path = self.model_path_input.text().strip() or "yolov8n.pt"
        standard_models = {
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"
        }
        if model_path not in standard_models and not os.path.exists(model_path):
            QMessageBox.critical(
                self,
                self.t("dialog_error_title"),
                self.t("err_model_not_found", model=model_path)
            )
            return

        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.results_data = {}
        self.image_selector.blockSignals(True)
        self.image_selector.clear()
        self.image_selector.blockSignals(False)
        self.log(self.t("log_starting"))

        needs_reload = (
            self.inference_engine is None
            or getattr(self.inference_engine, "model_path", None) != model_path
        )

        if needs_reload:
            try:
                self.log(self.t("log_loading_model", model=model_path))
                self.inference_engine = InferenceEngine(model_path)
            except Exception as e:
                self.log(self.t("log_model_error", error=str(e)))
                self.btn_run.setEnabled(True)
                return

        self.worker = WorkerThread(self.inference_engine, input_dir, conf)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, current, total, path, res_data):
        percent = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        img_name = os.path.basename(path)
        det_count = len(res_data.get("annotations", []))
        self.log(self.t("log_progress", current=current, total=total, name=img_name, count=det_count))

        if res_data:
            self.results_data[img_name] = res_data
            self.image_selector.blockSignals(True)
            self.image_selector.addItem(img_name)
            self.image_selector.setCurrentText(img_name)
            self.image_selector.blockSignals(False)
            self.update_preview(img_name, res_data)

    def on_finished(self, results):
        self.results_data = results
        output_dir = self.output_path_input.text().strip()
        export_format = self.format_combo.currentText()

        self.log(self.t("log_saving"))

        if export_format == "YOLO TXT":
            AnnotationExporter.to_yolo(results, output_dir)
        elif export_format == "LabelMe JSON":
            AnnotationExporter.to_labelme(results, output_dir)
        elif export_format == "COCO JSON":
            AnnotationExporter.to_coco(results, output_dir)

        self.btn_run.setEnabled(True)
        self.log(self.t("log_done", output_dir=output_dir))
        QMessageBox.information(self, self.t("dialog_success_title"), self.t("msg_done"))

    def on_error(self, err_msg):
        self.btn_run.setEnabled(True)
        self.log(self.t("log_error", error=err_msg))
        QMessageBox.critical(
            self,
            self.t("dialog_error_title"),
            self.t("err_processing", error=err_msg)
        )