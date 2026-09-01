import os
from pathlib import Path

import cv2

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient


# =====================================
# PROJECT DIRECTORY
# =====================================

BASE_DIR = Path(__file__).resolve().parent


# =====================================
# LOAD API KEY
# =====================================

load_dotenv(
    BASE_DIR / ".env"
)


API_KEY = os.getenv(
    "ROBOFLOW_API_KEY"
)


if not API_KEY:

    raise ValueError(
        "ROBOFLOW_API_KEY not found"
    )


API_KEY = API_KEY.strip()


print(
    "Roboflow API key loaded successfully."
)


# =====================================
# CREATE ROBOFLOW CLIENT
# =====================================

CLIENT = InferenceHTTPClient(

    api_url="https://serverless.roboflow.com",

    api_key=API_KEY

)


# =====================================
# MODEL ID
# =====================================

MODEL_ID = "veg1-hcqsf-2/4"


# =====================================
# IMAGE PATH
# =====================================

IMAGE_PATH = (
    BASE_DIR /
    "test_image" /
    "Onion07060_jpg.rf.PhNY2oC8HGrtlsbxi9D9.jpg"
)


# =====================================
# DISPLAY INFORMATION
# =====================================

print("\n")
print("=" * 55)

print(
    "TESTING ROBOFLOW MODEL"
)

print("=" * 55)


print(
    f"\nModel ID: {MODEL_ID}"
)


print(
    f"Image path: {IMAGE_PATH}"
)


print(
    f"Image exists: {IMAGE_PATH.exists()}"
)


# =====================================
# CHECK FILE
# =====================================

if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


# =====================================
# LOAD IMAGE WITH OPENCV
# =====================================

image = cv2.imread(
    str(IMAGE_PATH)
)


if image is None:

    raise ValueError(
        "OpenCV could not read the image."
    )


print(
    f"\nImage shape: {image.shape}"
)


print(
    f"Image dtype: {image.dtype}"
)


# =====================================
# RUN INFERENCE
# =====================================

try:

    print(
        "\nSending image to Roboflow..."
    )


    result = CLIENT.infer(

        image,

        model_id=MODEL_ID

    )


    print("\n")
    print("=" * 55)

    print(
        "INFERENCE SUCCESS"
    )

    print("=" * 55)


    print(
        "\nResult:"
    )


    print(
        result
    )


except Exception as e:

    print("\n")
    print("=" * 55)

    print(
        "INFERENCE FAILED"
    )

    print("=" * 55)


    print(
        "\nException type:"
    )


    print(
        type(e).__name__
    )


    print(
        "\nFull error:"
    )


    print(
        repr(e)
    )
