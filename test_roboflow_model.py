import cv2

from roboflow_grading import (
    analyze_onions
)


# ==========================================
# IMAGE PATH
# ==========================================

IMAGE_PATH = (
    r"test_image\Onion07060_jpg.rf.PhNY2oC8HGrtlsbxi9D9.jpg"
)


# ==========================================
# ANALYZE
# ==========================================

print("\n")
print("=" * 50)
print("ROBOFLOW ONION GRADING TEST")
print("=" * 50)


annotated_image, onion_data, raw_result = (
    analyze_onions(
        IMAGE_PATH
    )
)


# ==========================================
# PRINT RESULTS
# ==========================================

print(
    f"\nTotal objects detected: "
    f"{len(onion_data)}"
)


for onion in onion_data:


    print("\n")

    print(
        f"Onion: "
        f"{onion['Onion Number']}"
    )


    print(
        f"Model Detection: "
        f"{onion['Model Detection']}"
    )


    print(
        f"Final Condition: "
        f"{onion['Condition']}"
    )


    print(
        f"Confidence: "
        f"{onion['Confidence']}"
    )


# ==========================================
# SAVE RESULT
# ==========================================

OUTPUT_PATH = (
    "roboflow_result.jpg"
)


cv2.imwrite(
    OUTPUT_PATH,
    annotated_image
)


print("\n")
print("=" * 50)

print(
    f"Result saved to: "
    f"{OUTPUT_PATH}"
)

print("=" * 50)
