from ultralytics import YOLO
from pathlib import Path
import cv2
import csv
import json

from onion_analysis import analyze_onion

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DETECTOR_PATH = (
    BASE_DIR
    / "models"
    / "onion_detector_best.pt"
)

QUALITY_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "onion_quality_best.pt"
)

IMAGE_PATH = (
    BASE_DIR
    / "test"
    / "images"
    / "Onion13302_jpg.rf.N4qU4uSIDGkihwuf7Kgq.jpg"
)

# ============================================================
# SETTINGS
# ============================================================

DETECTION_CONFIDENCE = 0.25

# ============================================================
# LOAD MODELS
# ============================================================

print("======================================")
print("       ONION QUALITY INSPECTOR")
print("======================================")

print("\nLoading onion detector...")

detector = YOLO(
    str(DETECTOR_PATH)
)

print("Detector loaded.")

print("\nLoading quality classifier...")

quality_model = YOLO(
    str(QUALITY_MODEL_PATH)
)

print("Quality classifier loaded.")


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:

    raise FileNotFoundError(
        f"Could not load image:\n{IMAGE_PATH}"
    )


# ============================================================
# YOLO DETECTION
# ============================================================

print("\nRunning onion detection...")

results = detector(
    str(IMAGE_PATH),
    conf=DETECTION_CONFIDENCE,
    verbose=False
)

result = results[0]


# ============================================================
# STORAGE
# ============================================================

onion_data = []


# ============================================================
# PROCESS EACH ONION
# ============================================================

for i, box in enumerate(result.boxes):

    # --------------------------------------------------------
    # Detection confidence
    # --------------------------------------------------------

    detection_confidence = float(
        box.conf[0]
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    # Keep coordinates inside image
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(
        image.shape[1],
        x2
    )

    y2 = min(
        image.shape[0],
        y2
    )

    # --------------------------------------------------------
    # Crop onion
    # --------------------------------------------------------

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        continue


    # ========================================================
    # HEALTHY / UNHEALTHY CLASSIFICATION
    # ========================================================

    quality_results = quality_model(
        crop,
        verbose=False
    )

    quality_result = quality_results[0]

    quality_class_id = int(
        quality_result.probs.top1
    )

    quality_confidence = float(
        quality_result.probs.top1conf
    )

    quality_label = (
        quality_result.names[
            quality_class_id
        ]
    )


    # ========================================================
    # OPENCV ANALYSIS
    # ========================================================

    analysis = analyze_onion(
        crop,
        debug=False
    )


    # ========================================================
    # STORE INFORMATION
    # ========================================================

    onion_info = {

        "id": i + 1,

        "detection_confidence":
            round(
                detection_confidence,
                3
            ),

        "condition":
            quality_label,

        "condition_confidence":
            round(
                quality_confidence,
                3
            ),

        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    }


    # --------------------------------------------------------
    # Add OpenCV measurements
    # --------------------------------------------------------

    if analysis:

        onion_info.update({

            "analysis_method":
                analysis["method"],

            "area_px":
                analysis["area"],

            "width_px":
                analysis["width"],

            "height_px":
                analysis["height"],

            "aspect_ratio":
                analysis["aspect_ratio"],

            "circularity":
                analysis["circularity"],

            "brightness":
                analysis["mean_brightness"],

            "hue":
                analysis["mean_hue"],

            "saturation":
                analysis["mean_saturation"]
        })

    else:

        onion_info.update({

            "analysis_method":
                "FAILED",

            "area_px": None,

            "width_px": None,

            "height_px": None,

            "aspect_ratio": None,

            "circularity": None,

            "brightness": None,

            "hue": None,

            "saturation": None
        })


    onion_data.append(
        onion_info
    )


    # ========================================================
    # DRAW RESULT
    # ========================================================

    if quality_label.lower() == "healthy":

        label = (
            f"#{i + 1} "
            f"HEALTHY "
            f"{quality_confidence:.2f}"
        )

    else:

        label = (
            f"#{i + 1} "
            f"UNHEALTHY "
            f"{quality_confidence:.2f}"
        )


    # Bounding box
    cv2.rectangle(

        image,

        (x1, y1),

        (x2, y2),

        (0, 255, 0),

        2
    )


    # Label
    cv2.putText(

        image,

        label,

        (x1, max(25, y1 - 10)),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 0),

        2
    )


# ============================================================
# BATCH STATISTICS
# ============================================================

total = len(
    onion_data
)

healthy_count = sum(

    1
    for onion in onion_data

    if onion["condition"].lower()
    == "healthy"
)

unhealthy_count = sum(

    1
    for onion in onion_data

    if onion["condition"].lower()
    == "unhealthy"
)


if total > 0:

    healthy_percentage = (
        healthy_count
        / total
        * 100
    )

    unhealthy_percentage = (
        unhealthy_count
        / total
        * 100
    )

else:

    healthy_percentage = 0

    unhealthy_percentage = 0


# ============================================================
# AVERAGE MEASUREMENTS
# ============================================================

valid_analysis = [

    onion
    for onion in onion_data

    if onion["area_px"] is not None
]


if valid_analysis:

    avg_circularity = sum(

        onion["circularity"]
        for onion in valid_analysis

    ) / len(valid_analysis)


    avg_brightness = sum(

        onion["brightness"]
        for onion in valid_analysis

    ) / len(valid_analysis)

else:

    avg_circularity = 0

    avg_brightness = 0


# ============================================================
# TERMINAL REPORT
# ============================================================

print("\n")
print("======================================")
print("        ONION BATCH REPORT")
print("======================================")

print(
    f"Total onions detected : {total}"
)

print(
    f"Healthy               : "
    f"{healthy_count} "
    f"({healthy_percentage:.1f}%)"
)

print(
    f"Unhealthy             : "
    f"{unhealthy_count} "
    f"({unhealthy_percentage:.1f}%)"
)

print(
    f"Average circularity   : "
    f"{avg_circularity:.3f}"
)

print(
    f"Average brightness    : "
    f"{avg_brightness:.2f}"
)

print("======================================")


# ============================================================
# INDIVIDUAL REPORT
# ============================================================

print("\nINDIVIDUAL ONIONS")
print("--------------------------------------")

for onion in onion_data:

    print(
        f"\nOnion #{onion['id']}"
    )

    print(
        f"Detection confidence: "
        f"{onion['detection_confidence']}"
    )

    print(
        f"Condition: "
        f"{onion['condition']}"
    )

    print(
        f"Condition confidence: "
        f"{onion['condition_confidence']}"
    )

    if onion["area_px"] is not None:

        print(
            f"Area: "
            f"{onion['area_px']} px²"
        )

        print(
            f"Circularity: "
            f"{onion['circularity']}"
        )

        print(
            f"Brightness: "
            f"{onion['brightness']}"
        )


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    BASE_DIR
    / "batch_report.csv"
)


csv_columns = [

    "id",

    "detection_confidence",

    "condition",

    "condition_confidence",

    "x1",
    "y1",
    "x2",
    "y2",

    "analysis_method",

    "area_px",

    "width_px",

    "height_px",

    "aspect_ratio",

    "circularity",

    "brightness",

    "hue",

    "saturation"
]


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_columns
    )

    writer.writeheader()

    writer.writerows(
        onion_data
    )


# ============================================================
# SAVE JSON
# ============================================================

json_path = (
    BASE_DIR
    / "batch_report.json"
)


batch_report = {

    "total_onions":
        total,

    "healthy":
        healthy_count,

    "unhealthy":
        unhealthy_count,

    "healthy_percentage":
        round(
            healthy_percentage,
            2
        ),

    "unhealthy_percentage":
        round(
            unhealthy_percentage,
            2
        ),

    "average_circularity":
        round(
            avg_circularity,
            3
        ),

    "average_brightness":
        round(
            avg_brightness,
            2
        ),

    "onions":
        onion_data
}


with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        batch_report,
        f,
        indent=4
    )


# ============================================================
# SAVE ANNOTATED IMAGE
# ============================================================

output_path = (
    BASE_DIR
    / "phase1b_result.jpg"
)


cv2.imwrite(
    str(output_path),
    image
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n======================================")

print(
    "Result image:"
)

print(
    output_path
)

print(
    "\nCSV report:"
)

print(
    csv_path
)

print(
    "\nJSON report:"
)

print(
    json_path
)

print("======================================")
