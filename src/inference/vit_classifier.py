
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "slmci_vit_81k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ViTClassifier:
    def __init__(self):
        config_file = MODEL_PATH / "config.json"
        weights_file = MODEL_PATH / "model.safetensors"

        if not config_file.exists() or not weights_file.exists():
            raise FileNotFoundError(
                "ViT model files are missing!\n"
                f"Expected folder: {MODEL_PATH}\n"
                "Required: config.json and model.safetensors"
            )

        print("Loading 81K ViT model...")

        self.model = (
            AutoModelForImageClassification.from_pretrained(
                str(MODEL_PATH),
                local_files_only=True
            )
            .to(DEVICE)
        )

        self.model.eval()

        # Same preprocessing as our Colab training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("Model loaded successfully!")
        print("Device:", DEVICE)

    def predict(self, image_path):
        image_path = Path(image_path)

        if not image_path.is_absolute():
            image_path = ROOT / image_path

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        inputs = self.transform(image)
        inputs = inputs.unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            outputs = self.model(pixel_values=inputs)

            probabilities = torch.softmax(
                outputs.logits, dim=1
            )[0]

        scores, indices = torch.topk(probabilities, k=3)

        results = []

        for score, index in zip(scores, indices):
            label = self.model.config.id2label[
                index.item()
            ]

            results.append({
                "label": label,
                "confidence": score.item() * 100
            })

        return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.inference.vit_classifier "
            "A1.jpg data\\results\\hand_crop_1.jpg"
        )
        sys.exit(1)

    classifier = ViTClassifier()

    for image_path in sys.argv[1:]:
        print(f"\nImage: {image_path}")

        predictions = classifier.predict(image_path)

        print("Top 3 predictions:")

        for prediction in predictions:
            print(
                f"  {prediction['label']}: "
                f"{prediction['confidence']:.2f}%"
            )

    print("\nPrediction test completed!")
