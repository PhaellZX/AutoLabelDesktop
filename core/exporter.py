import os
import json
from datetime import datetime

class AnnotationExporter:
    @staticmethod
    def to_yolo(results, output_dir):
        """
        Exporta as anotações no formato YOLO TXT.
        Formato por linha no txt: <class_id> <x_center> <y_center> <width> <height> (valores normalizados de 0 a 1)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for img_name, data in results.items():
            width = data["width"]
            height = data["height"]
            txt_filename = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_filename)
            
            lines = []
            for ann in data["annotations"]:
                cls_id = ann["class_id"]
                x1, y1, x2, y2 = ann["bbox"]
                
                # Converte coordenadas absolutas (xyxy) para YOLO (centro e tamanho normalizados)
                box_width = x2 - x1
                box_height = y2 - y1
                x_center = x1 + (box_width / 2.0)
                y_center = y1 + (box_height / 2.0)
                
                # Normalização
                x_center /= width
                y_center /= height
                box_width /= width
                box_height /= height
                
                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    @staticmethod
    def to_labelme(results, output_dir):
        """
        Exporta no formato LabelMe JSON individual por imagem.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for img_name, data in results.items():
            json_filename = os.path.splitext(img_name)[0] + ".json"
            json_path = os.path.join(output_dir, json_filename)
            
            shapes = []
            for ann in data["annotations"]:
                x1, y1, x2, y2 = ann["bbox"]
                shapes.append({
                    "label": ann["class_name"],
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                })
                
            labelme_data = {
                "version": "5.0.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": img_name,
                "imageData": None,
                "imageHeight": data["height"],
                "imageWidth": data["width"]
            }
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(labelme_data, f, indent=4)

    @staticmethod
    def to_coco(results, output_dir):
        """
        Exporta em um arquivo único no formato COCO JSON padrão.
        """
        os.makedirs(output_dir, exist_ok=True)
        coco_path = os.path.join(output_dir, "annotations_coco.json")
        
        images = []
        annotations = []
        categories_map = {}
        
        ann_id_counter = 1
        
        for img_idx, (img_name, data) in enumerate(results.items(), start=1):
            images.append({
                "id": img_idx,
                "file_name": img_name,
                "width": data["width"],
                "height": data["height"]
            })
            
            for ann in data["annotations"]:
                cls_id = ann["class_id"]
                cls_name = ann["class_name"]
                
                if cls_id not in categories_map:
                    categories_map[cls_id] = cls_name
                
                x1, y1, x2, y2 = ann["bbox"]
                w = x2 - x1
                h = y2 - y1
                
                annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_idx,
                    "category_id": cls_id,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "segmentation": [],
                    "iscrowd": 0
                })
                ann_id_counter += 1
                
        categories = [{"id": cid, "name": cname, "supercategory": "none"} for cid, cname in categories_map.items()]
        
        coco_output = {
            "info": {
                "description": "AutoLabelDesktop Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().strftime("%Y-%m-%d")
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        with open(coco_path, "w", encoding="utf-8") as f:
            json.dump(coco_output, f, indent=4)