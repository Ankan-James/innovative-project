
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
from fastapi.staticfiles import StaticFiles

from starlette.concurrency import run_in_threadpool

from uuid import UUID
from fastapi import Form
from src.inference.sign_interpreter import SignInterpreter

from src.camera.video_recorder import (
    receive_video,
    delete_temporary_video
)
from src.camera.frame_sampler import sample_video
from src.inference.temporal_voting import TemporalVoting

from src.inference.autocorrect import AutoCorrect

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


# ========================================
# WORD BUILDING - PER CAMERA SESSION
# ========================================

word_sessions = {}
session_last_seen = {}
word_lock = threading.Lock()
autocorrect = AutoCorrect()

SESSION_TIMEOUT = 1800  # 30 minutes




def build_text(session_id, prediction, name_mode=False):
    """Build text and autocorrect completed words."""

    try:
        session_id = str(UUID(session_id))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid session ID"
        )

    with word_lock:
        now = time.monotonic()

        for sid, last_seen in list(session_last_seen.items()):
            if now - last_seen > SESSION_TIMEOUT:
                word_sessions.pop(sid, None)
                session_last_seen.pop(sid, None)

        if session_id not in word_sessions:
            word_sessions[session_id] = SignInterpreter()

        session_last_seen[session_id] = now

        interpreter = word_sessions[session_id]
        previous_text = interpreter.text

        text_result = interpreter.update(
            prediction["label"],
            hand_detected=prediction["hand_detected"]
        )

        current_text = text_result["text"]
        correction_result = None

        # Autocorrect only after a new SPACE sign.
        word_completed = (
            text_result["accepted"]
            and current_text.endswith(" ")
            and not previous_text.endswith(" ")
            and current_text != previous_text
        )

        if word_completed:
            words = current_text.strip().split()

            if words:
                completed_word = words[-1]

                correction_result = autocorrect.check_word(
                    completed_word,
                    name_mode=name_mode
                )

                if correction_result["status"] == "applied":
                    interpreter.text = (
                        current_text[:-(len(completed_word) + 1)]
                        + correction_result["corrected"]
                        + " "
                    )

                    text_result = interpreter.get_result(
                        text_result["accepted"]
                    )

    response = {
        **prediction,
        **text_result,
        "raw_label": text_result.get("raw_label") or prediction.get("label"),
        "accepted_label": text_result.get("accepted_label"),
        "adaptive_corrected": text_result.get("adaptive_corrected", False),
        "autocorrect": correction_result
    }
    if text_result.get("label") is not None:
        response["label"] = text_result["label"]

    return response


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

def detect_hand(image, return_landmarks=False):
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
        return (None, None, None) if return_landmarks else (None, None)

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

    bbox = [x1, y1, x2, y2]

    if return_landmarks:
        landmarks = [
            (point.x, point.y, point.z)
            for point in hand
        ]
        return crop, bbox, landmarks

    return crop, bbox


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


# ========================================
# VIDEO RECOGNITION: 8-FRAME VOTING
# ========================================


def analyze_video(video_path):

    
    start_time = time.perf_counter()

    # Extract 8 frames from the video
    frames = sample_video(video_path, num_frames=8)
    
    frame_width, frame_height = frames[-1].size
    last_bbox = None

    voting = TemporalVoting(
        window_size=8,
        min_votes=5,
        min_confidence=0.50
    )

    BATCH_SIZE = 4

    # ========================================
    # STEP 1: DETECT AND CROP HANDS
    # ========================================

    hand_tensors = []

    for frame_index, frame in enumerate(frames):
        hand_crop, bbox = detect_hand(frame)

        # Use the final sampled frame for the hand box
        if frame_index == len(frames) - 1:
            last_bbox = bbox

        if hand_crop is not None:
            tensor = preprocess(hand_crop)
            hand_tensors.append(tensor)

    detection_time = time.perf_counter()

    hands_detected = len(hand_tensors)

    if hands_detected == 0:
        print("No hands detected.")

        return {
            "hand_detected": False,
            "label": None,
            "confidence": 0,
            "votes": 0,
            "frames_analyzed": len(frames),
            "message": "No hand detected",
            "bbox": last_bbox,
            "frame_width": frame_width,
            "frame_height": frame_height,
        }

    # ========================================
    # STEP 2: BATCH INFERENCE (4 AT A TIME)
    # ========================================

    predictions = []
    stable_label = None

    for start in range(0, hands_detected, BATCH_SIZE):

        batch_tensors = hand_tensors[
            start:start + BATCH_SIZE
        ]

        batch = torch.stack(batch_tensors)

        with torch.inference_mode():
            outputs = model(pixel_values=batch)
            probabilities = torch.softmax(
                outputs.logits, dim=1
            )

        confidences, predicted_ids = torch.max(
            probabilities, dim=1
        )

        # ====================================
        # STEP 3: TEMPORAL VOTING
        # ====================================

        for predicted_id, confidence in zip(
            predicted_ids.tolist(),
            confidences.tolist()
        ):

            label = CLASS_NAMES[predicted_id]

            if label == "NOTHING":
                predictions.clear()

            elif confidence >= 0.50:
                predictions.append(
                    (label, confidence)
                )

            result = voting.update(
                label, confidence
            )

            if result is not None:
                stable_label = result

    inference_time = time.perf_counter()

    # ========================================
    # STEP 4: FINAL RESULT
    # ========================================

    history = voting.get_history()

    print("\nSLMCI Performance:")
    print(
        f"Hand processing: "
        f"{detection_time - start_time:.2f}s"
    )
    print(
        f"ViT inference + voting: "
        f"{inference_time - detection_time:.2f}s"
    )
    print(
        f"Total backend processing: "
        f"{inference_time - start_time:.2f}s"
    )

    if (
        stable_label is None
        or history.count(stable_label) < 5
    ):
        return {
            "hand_detected": True,
            "label": None,
            "confidence": 0,
            "votes": 0,
            "frames_analyzed": len(frames),
            "message": "Not enough matching predictions",
            "bbox": last_bbox,
            "frame_width": frame_width,
            "frame_height": frame_height,
        }

    matching_scores = [
        confidence
        for label, confidence in predictions
        if label == stable_label
    ]

    average_confidence = (
        sum(matching_scores) / len(matching_scores)
    )

    return {
        "hand_detected": True,
        "label": stable_label,
        "confidence": round(
            average_confidence * 100, 2
        ),
        "votes": history.count(stable_label),
        "frames_analyzed": len(frames),
        "message": "Prediction complete",
        "bbox": last_bbox,
        "frame_width": frame_width,
        "frame_height": frame_height,
    }


@app.post("/predict-video")
async def predict_video(
    video: UploadFile = File(...),
    session_id: str = Form(...),
    name_mode: bool = Form(False)
):
    """
    Receive a short webcam recording and
    return its final sign prediction.
    """

    video_path = None

    try:
        video_path = await receive_video(
            video,
            save_permanently=False
        )

        result = await run_in_threadpool(
            analyze_video,
            video_path
        )

        return build_text(session_id, result, name_mode=name_mode)

    except (ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    finally:
        if video_path is not None:
            delete_temporary_video(video_path)


# ========================================
# STATIC FILES SERVING
# ========================================

app.mount("/css", StaticFiles(directory=ROOT / "web" / "css"), name="css")
app.mount("/js", StaticFiles(directory=ROOT / "web" / "js"), name="js")
app.mount("/components", StaticFiles(directory=ROOT / "web" / "components"), name="components")
app.mount("/assets", StaticFiles(directory=ROOT / "web" / "assets"), name="assets")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

