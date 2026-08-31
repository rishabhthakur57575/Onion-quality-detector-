from ultralytics import YOLO
from pathlib import Path
import cv2

# -----------------------------------------
# Paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"
IMAGE_PATH = BASE_DIR / "test" / "images" / "Onion13299_jpg.rf.yetvn85PjoYJCLVckXB9.jpg"


# -----------------------------------------
# Load model
# -----------------------------------------

print("Loading model...")

model = YOLO(str(MODEL_PATH))


# -----------------------------------------
# Run YOLO
# -----------------------------------------

results = model(
    str(IMAGE_PATH),
    conf=0.25
)

result = results[0]


# -----------------------------------------
# Count onions
# -----------------------------------------

count = len(result.boxes)


print("\n==============================")
print("YOLO ONION DETECTION")
print("==============================")

print("Detected onions:", count)


# -----------------------------------------
# Print detections
# -----------------------------------------

for i, box in enumerate(result.boxes):

    confidence = float(box.conf[0])

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    print(
        f"Onion {i + 1}: "
        f"confidence={confidence:.2f}, "
        f"box=({x1},{y1},{x2},{y2})"
    )


# -----------------------------------------
# Save result
# -----------------------------------------

output_path = BASE_DIR / "yolo_result.jpg"

result.save(
    filename=str(output_path)
)

print("\nSaved result:")
print(output_path)
