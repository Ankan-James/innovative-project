
import base64
import io
import os
import time
import threading
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse


# ========================================
# CONFIGURATION
# ========================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "slmci_vit_81k"
HAND_PATH = ROOT / "models" / "hand_landmarker.task"

CLASS_NAMES = list("ABCDEFGHIKLMNOPQRSTUVWXY") + [
    "SPACE", "DEL", "NOTHING"
]

app = FastAPI(title="SLMCI Sign Language API")

torch.set_num_threads(min(4, os.cpu_count() or 2))


# ========================================
# LOAD EXISTING TRAINED VIT
# ========================================

if not (MODEL_PATH / "config.json").exists():
    raise FileNotFoundError(f"Missing config.json in {MODEL_PATH}")

if not (MODEL_PATH / "model.safetensors").exists():
    raise FileNotFoundError(f"Missing model.safetensors in {MODEL_PATH}")

if not HAND_PATH.exists():
    raise FileNotFoundError(f"Missing MediaPipe model: {HAND_PATH}")

print("Loading SLMCI ViT model...")

model = AutoModelForImageClassification.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
).to("cpu")

model.eval()

if model.config.num_labels != len(CLASS_NAMES):
    raise RuntimeError(
        f"Expected 27 classes, got {model.config.num_labels}"
    )

print("ViT model loaded successfully!")


# ========================================
# PREPROCESSING FROM COLAB TRAINING
# ========================================

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ========================================
# MEDIAPIPE VIDEO TRACKING
# ========================================

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(HAND_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.35,
    min_hand_presence_confidence=0.35,
    min_tracking_confidence=0.35
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(
    options
)

detector_lock = threading.Lock()
last_timestamp = -1

print("MediaPipe video hand tracker ready!")


# ========================================
# HAND DETECTION AND SQUARE CROPPING
# ========================================

def detect_hand(image):
    global last_timestamp

    width, height = image.size
    rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Video mode requires increasing timestamps.
    # Lock makes this safe for concurrent requests.
    with detector_lock:
        now = time.monotonic_ns() // 1_000_000
        timestamp = max(now, last_timestamp + 1)
        last_timestamp = timestamp

        result = detector.detect_for_video(
            mp_image,
            timestamp
        )

    if not result.hand_landmarks:
        return None, None

    hand = result.hand_landmarks[0]

    xs = [int(p.x * width) for p in hand]
    ys = [int(p.y * height) for p in hand]

    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(width, max(xs))
    y2 = min(height, max(ys))

    if x2 <= x1 or y2 <= y1:
        return None, None

    # Keep the hand proportions: use a square crop.
    hand_size = max(x2 - x1, y2 - y1)
    side = max(80, int(hand_size * 1.8))
    side = min(side, width, height)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    left = int(center_x - side / 2)
    top = int(center_y - side / 2)

    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))

    crop = image.crop((
        left,
        top,
        left + side,
        top + side
    ))

    return crop, [x1, y1, x2, y2]


# ========================================
# ROUTES
# ========================================

@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ready",
        "model": "slmci_vit_81k",
        "hand_detection": "mediapipe_video"
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):

    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "JPEG or PNG required")

    raw = file.file.read(5 * 1024 * 1024 + 1)

    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(413, "Image too large")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    hand_crop, bbox = detect_hand(image)

    if hand_crop is None:
        return {
            "hand_detected": False,
            "label": None,
            "confidence": 0,
            "bbox": None,
            "crop_preview": None
        }

    # Predict using our existing 81K model
    tensor = preprocess(hand_crop).unsqueeze(0)

    with torch.inference_mode():
        logits = model(pixel_values=tensor).logits
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_id = int(probabilities.argmax().item())
    score = float(probabilities[predicted_id].item())

    label = CLASS_NAMES[predicted_id]

    # Send a preview of the actual model input crop.
    preview = hand_crop.resize((224, 224))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=80)

    preview_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return {
        "hand_detected": True,
        "label": label,
        "confidence": round(score * 100, 2),
        "bbox": bbox,
        "crop_preview": preview_base64
    }
