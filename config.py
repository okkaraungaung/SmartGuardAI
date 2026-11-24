import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = "smartguard-dev-key"
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "videos")
    FACES_FOLDER = os.path.join(BASE_DIR, "uploads", "faces")
    PROCESSED_FOLDER = os.path.join(BASE_DIR, "results", "processed_videos")
    SNAPSHOT_FOLDER = os.path.join(BASE_DIR, "results", "snapshots")
    DATABASE_PATH = os.path.join(BASE_DIR, "database", "smartguard.db")

    ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}
    ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024
    FRAME_SAMPLE_INTERVAL = 5

    # Phase 2: Image Processing
    TARGET_WIDTH = 640
    GAUSSIAN_KERNEL = (5, 5)
    GAMMA_VALUE = 1.8
    CLAHE_CLIP_LIMIT = 2.5
    CLAHE_TILE_GRID = (8, 8)
    LOW_LIGHT_THRESHOLD = 85.0

    # Phase 3: YOLO Object Detection (Person & Package-like objects only)
    YOLO_MODEL_PATH = "yolov8n.pt"
    DETECTION_CONFIDENCE = 0.45
    TARGET_CLASSES = {
        0: "person",
        24: "backpack",
        26: "handbag",
        28: "suitcase"
    }

    # Phase 4: Face Recognition
    YUNET_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
    SFACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_recognition_sface_2021dec.onnx")
    FACE_SIMILARITY_THRESHOLD = 0.38

    # Phase 5: Security Events & Telegram Defaults
    DEFAULT_LOITERING_SECONDS = 15.0
    EVENT_COOLDOWN_SECONDS = 60.0
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")