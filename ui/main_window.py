import os
import threading
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QComboBox,
    QProgressBar, QTextEdit, QGroupBox, QMessageBox
)

from core.inference import InferenceEngine
from core.exporter import AnnotationExporter

class WorkerThread(QThread):
    """
    Thread separada para rodar a inferência sem travar a interface gráfica.
    """
    progress = Signal(int, int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, engine, input_dir, conf):
        super().__init__()
        self.engine = engine
        self.input_dir = input_dir
        self.conf = conf

    def run(self):
        try:
            # Sobrescreve temporariamente o limite de confiança se necessário
            results = self.engine.batch_process(
                self.input_dir, 
                output_callback=lambda curr, total, path: self.progress.emit(curr, total, path)
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoLabel Desktop - AI Annotation Studio")
        self.resize(700, 500)
        
        self.inference_engine = None
        self.results_data = None
        
        self.init_ui()
        self.init_engine_async()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Grupo 1: Seleção de Pastas
        dir_group = QGroupBox("Diretórios de Trabalho")
        dir_layout = QVBoxLayout()
        
        self.input_path_input = QLineEdit()
        self.input_path_input.setPlaceholderText("Selecione a pasta com as imagens...")
        btn_browse_input = QPushButton("Procurar Pasta de Imagens")
        btn_browse_input.clicked.connect(self.select_input_dir)
        
        input_h = QHBoxLayout()
        input_h.addWidget(self.input_path_input)
        input_h.addWidget(btn_browse_input)
        dir_layout.addLayout(input_h)

        self.output_path_input = QLineEdit()
        self.output_path_input.setPlaceholderText("Selecione a pasta de saída para as anotações...")
        btn_browse_output = QPushButton("Procurar Pasta de Saída")
        btn_browse_output.clicked.connect(self.select_output_dir)
        
        output_h = QHBoxLayout()
        output_h.addWidget(self.output_path_input)
        output_h.addWidget(btn_browse_output)
        dir_layout.addLayout(output_h)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Grupo 2: Configurações e Formato de Exportação
        settings_group = QGroupBox("Configurações de Processamento")
        settings_layout = QHBoxLayout()
        
        settings_layout.addWidget(QLabel("Formato de Saída:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["YOLO TXT", "LabelMe JSON", "COCO JSON"])
        settings_layout.addWidget(self.format_combo)
        
        settings_layout.addWidget(QLabel("Confiança Mínima:"))
        self.conf_input = QLineEdit("0.25")
        self.conf_input.setMaximumWidth(60)
        settings_layout.addWidget(self.conf_input)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Botão de Execução e Barra de Progresso
        self.btn_run = QPushButton("Iniciar Auto-Rotulagem por IA")
        self.btn_run.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.start_processing)
        layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Log Console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Logs e status do sistema aparecerão aqui...")
        layout.addWidget(self.console)

    def log(self, text):
        self.console.append(text)

    def init_engine_async(self):
        """Agora apenas avisa que o app está pronto, adiando o peso do YOLO para o clique"""
        self.log("[INFO] Interface pronta. O motor YOLO será inicializado ao iniciar o processamento.")

    def _load_engine_thread(self):
        try:
            self.inference_engine = InferenceEngine("yolov8n.pt")
            self.log(f"[SUCESSO] Modelo carregado na GPU/CPU com sucesso! Dispositivo: {self.inference_engine.device.upper()}")
        except Exception as e:
            self.log(f"[ERRO] Falha ao carregar modelo de IA: {str(e)}")

    def select_input_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Imagens")
        if dir_name:
            self.input_path_input.setText(dir_name)

    def select_output_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Saída")
        if dir_name:
            self.output_path_input.setText(dir_name)

    def start_processing(self):
        input_dir = self.input_path_input.text()
        output_dir = self.output_path_input.text()

        if not input_dir or not os.path.exists(input_dir):
            QMessageBox.critical(self, "Erro", "Selecione uma pasta de entrada válida.")
            return

        if not output_dir:
            QMessageBox.critical(self, "Erro", "Selecione uma pasta de saída válida.")
            return

        try:
            conf = float(self.conf_input.text())
        except ValueError:
            conf = 0.25

        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log(f"[INÍCIO] Preparando motor de inferência e processando imagens...")

        # Instancia o motor de forma segura na hora da execução
        if not self.inference_engine:
            try:
                self.inference_engine = InferenceEngine("yolov8n.pt")
            except Exception as e:
                self.log(f"[ERRO CRÍTICO] Falha ao carregar o modelo YOLO: {e}")
                self.btn_run.setEnabled(True)
                return

        # Inicia a thread de processamento em lote
        self.worker = WorkerThread(self.inference_engine, input_dir, conf)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, current, total, path):
        percent = int((current / total) * 100)
        self.progress_bar.setValue(percent)
        self.log(f"[{current}/{total}] Processado: {os.path.basename(path)}")

    def on_finished(self, results):
        self.results_data = results
        output_dir = self.output_path_input.text()
        export_format = self.format_combo.currentText()

        self.log("[INFO] Salvando anotações nos formatos selecionados...")

        if export_format == "YOLO TXT":
            AnnotationExporter.to_yolo(results, output_dir)
        elif export_format == "LabelMe JSON":
            AnnotationExporter.to_labelme(results, output_dir)
        elif export_format == "COCO JSON":
            AnnotationExporter.to_coco(results, output_dir)

        self.btn_run.setEnabled(True)
        self.log(f"[SUCESSO] Processamento concluído! Arquivos salvos em: {output_dir}")
        QMessageBox.information(self, "Sucesso", "Todas as imagens foram rotuladas e exportadas com sucesso!")

    def on_error(self, err_msg):
        self.btn_run.setEnabled(True)
        self.log(f"[ERRO CRÍTICO] {err_msg}")
        QMessageBox.critical(self, "Erro", f"Ocorreu um erro durante o processamento:\n{err_msg}")