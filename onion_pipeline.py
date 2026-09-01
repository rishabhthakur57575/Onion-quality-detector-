from pathlib import Path
import cv2

from roboflow_grading import analyze_onions


# =====================================
# CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR /
    "test" /
    "images" /
    "Onion13364_jpg.rf.Ulf8LXLw2grOxarqqnHX.jpg"
)


OUTPUT_PATH = (
    BASE_DIR /
    "roboflow_result.jpg"
)


# =====================================
# RUN ANALYSIS
# =====================================

print("\n")
print("=" * 55)
print("ONION GRADING PIPELINE")
print("=" * 55)


results = analyze_onions(
    IMAGE_PATH
)


predictions = results[
    "predictions"
]


# =====================================
# PRINT RESULTS
# =====================================

print(
    f"\nTotal objects detected: "
    f"{results['total_detected']}"
)


healthy_count = 0
unhealthy_count = 0
sprouted_count = 0
review_count = 0


for onion in predictions:

    condition = onion[
        "Condition"
    ]


    if condition == "Healthy":

        healthy_count += 1


    elif condition == "Unhealthy":

        unhealthy_count += 1


    elif condition == "Sprouted":

        sprouted_count += 1


    else:

        review_count += 1


    print("\n" + "-" * 40)

    print(
        f"Onion {onion['Onion Number']}"
    )

    print(
        f"Detected class: "
        f"{onion['Detected Class']}"
    )

    print(
        f"Condition: "
        f"{condition}"
    )

    print(
        f"Confidence: "
        f"{onion['Confidence']}"
    )


# =====================================
# SUMMARY
# =====================================

print("\n")
print("=" * 55)
print("BATCH SUMMARY")
print("=" * 55)

print(
    f"Total: {len(predictions)}"
)

print(
    f"Healthy: {healthy_count}"
)

print(
    f"Unhealthy: {unhealthy_count}"
)

print(
    f"Sprouted: {sprouted_count}"
)

print(
    f"Manual Review: {review_count}"
)


# =====================================
# SAVE IMAGE
# =====================================

cv2.imwrite(
    str(OUTPUT_PATH),
    results["annotated_image"]
)


print("\nResult saved to:")

print(
    OUTPUT_PATH
)
