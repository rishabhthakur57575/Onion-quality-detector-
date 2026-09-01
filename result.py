from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import os

# Change this to your classifier
model = YOLO(r"models/onion_quality_best.pt")

val_dir = r"quality_dataset_two\val"

class_names = {
    0: "healthy",
    1: "unhealthy"
}

y_true = []
y_pred = []

for class_id, class_name in class_names.items():

    folder = os.path.join(val_dir, class_name)

    for filename in os.listdir(folder):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        image_path = os.path.join(folder, filename)

        result = model(image_path, verbose=False)[0]

        predicted_class = result.probs.top1

        y_true.append(class_id)
        y_pred.append(predicted_class)


print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["healthy", "unhealthy"],
        digits=4
    )
)
