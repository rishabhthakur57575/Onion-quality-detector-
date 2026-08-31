from ultralytics import YOLO
from pathlib import Path


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent

    DATASET_DIR = (
        BASE_DIR
        / "quality_dataset"
    )

    print("======================================")
    print("ONION QUALITY CLASSIFIER")
    print("======================================")

    print("Dataset:")
    print(DATASET_DIR)

    # -----------------------------------------
    # Load pretrained classification model
    # -----------------------------------------

    model = YOLO("yolo26n-cls.pt")

    # -----------------------------------------
    # Train
    # -----------------------------------------

    model.train(

        data=str(DATASET_DIR),

        epochs=50,

        imgsz=224,

        batch=16,

        project=str(
            BASE_DIR / "runs"
        ),

        name="onion_quality_classifier",

        device="cpu",

        workers=4,

        patience=10,

        pretrained=True,

        verbose=True
    )

    print("\n======================================")
    print("QUALITY MODEL TRAINING COMPLETED")
    print("======================================")

    print(
        BASE_DIR
        / "runs"
        / "onion_quality_classifier"
        / "weights"
        / "best.pt"
    )
