import cv2
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", target_classes=None, conf_threshold=0.40):
        self.model = YOLO(model_path)
        self.target_classes = target_classes or {0: "person"}
        self.conf_threshold = conf_threshold

        self.colors = {
            "known_person": (0, 220, 0),     # Bright Green
            "unknown_person": (0, 80, 255),  # Red-Orange
            "vehicle": (255, 165, 0),        # Blue-Orange
            "package": (0, 215, 255)         # Yellow
        }

    def reset_tracker(self):
        """Resets the YOLO predictor and ByteTrack state cleanly before a new stream."""
        self.model.predictor = None

    def detect_and_track(self, frame):
        """
        Runs YOLOv8 + ByteTrack with a short 3-frame buffer so boxes vanish immediately when someone leaves.
        """
        results = self.model.track(
            source=frame,
            conf=max(self.conf_threshold, 0.50),
            classes=list(self.target_classes.keys()),
            persist=True,
            tracker="custom_bytetrack.yaml",
            verbose=False
        )

        detections = []
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = round(float(box.conf[0].item()), 3)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            track_id = int(box.id[0].item()) if box.id is not None else None
            label = self.target_classes.get(cls_id, "unknown")

            detections.append({
                "track_id": track_id,
                "class_id": cls_id,
                "label": label,
                "person_name": None,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        return detections
    def draw_detections(self, frame, detections):
        """
        Draws bounding boxes for People (with Name/Unknown + dwell timer) and Packages only.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            tid = det.get("track_id")
            person_name = det.get("person_name")
            dwell = det.get("dwell_time", 0.0)
            face_score = det.get("face_score", 0.0)

            if label == "person":
                display_name = person_name or "Scanning..."
                if display_name in ("Unknown", "Scanning..."):
                    color = self.colors["unknown_person"]
                    if display_name == "Unknown":
                        extra = f" ({dwell:.1f}s)" if det.get("in_roi", True) else " (Outside ROI)"
                    else:
                        extra = ""
                else:
                    color = self.colors["known_person"]
                    extra = f" ({face_score:.2f})" if face_score > 0 else ""
                id_prefix = f"ID #{tid} | " if tid is not None else ""
                text = f"{id_prefix}{display_name}{extra}"
            elif label in {"backpack", "handbag", "suitcase"}:
                color = self.colors["package"]
                text = f"Package ({label}) {det['confidence']:.2f}"
            else:
                continue

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated,
                text,
                (x1 + 3, max(y1 - 4, th + 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255) if label == "person" else (0, 0, 0),
                2
            )
        return annotated