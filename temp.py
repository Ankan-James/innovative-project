
from pathlib import Path
import shutil

# SLMCI project root
ROOT = Path(__file__).resolve().parent

# ==========================================
# 1. FOLDERS
# ==========================================

folders = [
    # Dataset
    "data/raw",
    "data/processed",
    "data/webcam_samples/labeled",
    "data/webcam_samples/unlabeled",
    "data/recordings",
    "data/results",

    # Models
    "models/slmci_vit_81k",
    "models/slmci_vit_27",

    # Source code
    "src/camera",
    "src/vision",
    "src/inference",
    "src/morse",
    "src/ui",
    "src/utils",

    # Training
    "training/notebooks",
    "training/scripts",

    # Testing and logs
    "tests",
    "logs",
]

# ==========================================
# 2. FILES
# ==========================================

files = [
    # Project root
    "main.py",
    "requirements.txt",
    ".gitignore",
    "README.md",

    # Main source
    "src/__init__.py",
    "src/pipeline.py",

    # Camera
    "src/camera/__init__.py",
    "src/camera/webcam.py",
    "src/camera/frame_sampler.py",
    "src/camera/video_recorder.py",

    # Computer vision
    "src/vision/__init__.py",
    "src/vision/hand_detector.py",
    "src/vision/hand_tracker.py",
    "src/vision/hand_cropper.py",
    "src/vision/preprocessing.py",

    # Model inference
    "src/inference/__init__.py",
    "src/inference/vit_classifier.py",
    "src/inference/rf_classifier.py",
    "src/inference/temporal_voting.py",
    "src/inference/sign_interpreter.py",

    # Morse code
    "src/morse/__init__.py",
    "src/morse/morse_encoder.py",
    "src/morse/morse_decoder.py",
    "src/morse/morse_input.py",

    # User interface
    "src/ui/__init__.py",
    "src/ui/webcam_display.py",
    "src/ui/overlays.py",

    # Utilities
    "src/utils/__init__.py",
    "src/utils/config.py",
    "src/utils/logger.py",

    # Training scripts
    "training/scripts/prepare_dataset.py",
    "training/scripts/capture_webcam_samples.py",
    "training/scripts/evaluate_model.py",

    # Tests
    "tests/test_hand_detection.py",
    "tests/test_preprocessing.py",
    "tests/test_vit.py",
    "tests/test_rf.py",
    "tests/test_temporal_voting.py",
    "tests/test_sign_interpreter.py",
    "tests/test_morse.py",
]

# ==========================================
# 3. CREATE FOLDERS
# ==========================================

print("\nCreating folders...\n")

for folder in folders:
    path = ROOT / folder

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"[CREATED] {folder}")
    else:
        print(f"[EXISTS]  {folder}")

# ==========================================
# 4. CREATE FILES
# ==========================================

print("\nCreating files...\n")

for file in files:
    path = ROOT / file

    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()
        print(f"[CREATED] {file}")
    else:
        print(f"[EXISTS]  {file}")

# ==========================================
# 5. BACK UP OLD WEBCAM PROGRAMS
# ==========================================

print("\nChecking old webcam programs...\n")

legacy_files = {
    "webcam_vit.py": "tests/legacy_webcam_vit.py",
    "webcam_rf.py": "tests/legacy_webcam_rf.py",
}

for old_name, new_name in legacy_files.items():
    source = ROOT / old_name
    destination = ROOT / new_name

    if source.is_file() and not destination.exists():
        shutil.copy2(source, destination)
        print(f"[COPIED] {old_name} -> {new_name}")

    elif destination.exists():
        print(f"[EXISTS] {new_name}")

    else:
        print(f"[SKIPPED] {old_name} not found")

# ==========================================
# 6. FINISHED
# ==========================================

print("\n" + "=" * 45)
print("SLMCI PROJECT STRUCTURE READY!")
print("=" * 45)

print("\nProject location:", ROOT)
print("Existing files were NOT overwritten.")
print("Trained models were NOT modified.")
print("Existing webcam scripts were preserved.")
print("\nYou can now start implementing the modules!")
