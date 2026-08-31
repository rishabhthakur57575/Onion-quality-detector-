from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent

    DATASET_YAML = BASE_DIR / "data.yaml"

    print("=" * 50)
    print("ONION YOLO TRAINING")
    print("=" * 50)

    print("\nUsing YAML:")
    print(DATASET_YAML)

    print("\nYAML CONTENT:")
    print(DATASET_YAML.read_text())

    print("=" * 50)

    model = YOLO("yolo26n.pt")

    model.train(
        data=str(DATASET_YAML),
        epochs=80,
        imgsz=640,
        batch=8,
        project=str(BASE_DIR / "runs"),
        name="onion_detector_v3",
        device="cpu",
        workers=4,
        patience=20,
        save=True,
        pretrained=True,
        verbose=True
    )
