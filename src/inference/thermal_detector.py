from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = Path("runs/detect/models/thermal/flir_thermal/weights/best.pt")


class ThermalDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None

        if self.model_path.exists():
            self.model = YOLO(str(self.model_path))

    def predict(self, image_path):
        if self.model is None:
            return {
                "available": False,
                "person_probability": 0.0,
                "person_count": 0,
                "detections": []
            }

        results = self.model.predict(
            source=str(image_path),
            conf=0.25,
            verbose=False
        )

        result = results[0]
        detections = []

        for box in result.boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])

            if cls == 0:
                xyxy = box.xyxy[0].tolist()

                detections.append({
                    "confidence": confidence,
                    "bbox": xyxy
                })

        person_probability = (
            max(d["confidence"] for d in detections)
            if detections else 0.0
        )

        return {
            "available": True,
            "person_probability": person_probability,
            "person_count": len(detections),
            "detections": detections
        }