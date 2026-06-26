import argparse
import torch

from src.utils import (
    load_model,
    predict_image,
    predict_topk
)

# ---------------------------------------
# Paths
# ---------------------------------------

CHECKPOINT_PATH = "checkpoints/best_model.pth"

# ---------------------------------------
# Device
# ---------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print(f"Using Device : {device}")
print("=" * 60)

# ---------------------------------------
# Load Model
# ---------------------------------------

model = load_model(
    CHECKPOINT_PATH,
    device
)

# ---------------------------------------
# Command Line Argument
# ---------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "image_path",
    type=str,
    help="Path to input image"
)

args = parser.parse_args()
# ---------------------------------------
# Prediction
# ---------------------------------------

prediction, confidence = predict_image(
    model,
    args.image_path,
    device
)

print()

print("=" * 60)
print("Prediction")
print("=" * 60)

print(f"Predicted Class : {prediction}")

print(f"Confidence      : {confidence*100:.2f}%")

print()

print("=" * 60)
print("Top 3 Predictions")
print("=" * 60)

top3 = predict_topk(
    model,
    args.image_path,
    device
)

for cls, score in top3:

    print(f"{cls:<20} {score*100:.2f}%")
