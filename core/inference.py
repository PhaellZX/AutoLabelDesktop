import os
import torch
from ultralytics import YOLO


class InferenceEngine:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Inicializa o motor de inferência de forma segura.
        """
        self.model_path = model_path
        self.device = "cpu"
        print(f"[INFO] Utilizando dispositivo: {self.device.upper()}")

        try:
            torch.set_num_threads(1)
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            print(f"[ERRO NO MOTOR] Falha ao instanciar o modelo YOLO: {e}")
            raise e

    def process_image(self, image_path, conf_threshold=0.25):
        if not os.path.exists(image_path):
            return None

        results = self.model(image_path, conf=conf_threshold, verbose=False)[0]
        height, width = results.orig_shape
        annotations = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            annotations.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox": [round(coord, 2) for coord in xyxy]
            })

        return {
            "image_path": image_path,
            "width": width,
            "height": height,
            "annotations": annotations
        }