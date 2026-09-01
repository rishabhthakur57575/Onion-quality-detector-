import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError(
        "ROBOFLOW_API_KEY not found in .env"
    )

API_KEY = API_KEY.strip()


MODEL_ID = "veg1-hcqsf-2/4"


# ==========================================
# ROBOFLOW CLIENT
# ==========================================

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)


# ==========================================
# SPROUT DETECTION SETTINGS
# ==========================================

# Lower this value to make sprout detection
# more sensitive.
#
# Current value is intentionally sensitive
# for your prototype.

SPROUT_GREEN_RATIO_THRESHOLD = 0.003


# ==========================================
# CLASS MAPPING
# ==========================================

def map_to_condition(model_class):

    model_class = str(model_class).lower().strip()

    # --------------------------------------
    # HEALTHY ONION
    # --------------------------------------

    if model_class == "onion":
        return "Healthy"


    # --------------------------------------
    # SPROUTED ONION
    # --------------------------------------

    elif model_class == "sprouted":
        return "Sprouted"


    # --------------------------------------
    # UNHEALTHY / DEFECTIVE ONIONS
    # --------------------------------------

    elif model_class in [

        "black smut",
        "black_smut",

        "double split",
        "double_split",

        "spoiled",

        "staining",

        "unhealthy",

        "manual review",
        "manual_review",

        "unknown"

    ]:

        return "Rotten"


    # --------------------------------------
    # EVERYTHING ELSE
    # --------------------------------------

    else:

        return "Rotten"


# ==========================================
# GET BOX COLOR
# ==========================================

def get_color(condition):

    if condition == "Healthy":

        return (0, 200, 0)


    elif condition == "Rotten":

        return (0, 0, 255)


    elif condition == "Sprouted":

        # Blue

        return (255, 0, 0)


    else:

        return (0, 0, 255)


# ==========================================
# SPROUT ANALYSIS
# ==========================================

def detect_sprout(image_roi):

    """
    Detect possible sprouting using OpenCV.

    Looks for green sprout-like pixels
    inside the detected onion bounding box.

    Returns:

        is_sprouted
        sprout_confidence
        green_ratio
    """

    if image_roi is None:

        return False, 0.0, 0.0


    if image_roi.size == 0:

        return False, 0.0, 0.0


    # --------------------------------------
    # RESIZE SMALL ROI
    # --------------------------------------

    height, width = image_roi.shape[:2]

    if height < 20 or width < 20:

        return False, 0.0, 0.0


    # --------------------------------------
    # CONVERT TO HSV
    # --------------------------------------

    hsv = cv2.cvtColor(
        image_roi,
        cv2.COLOR_BGR2HSV
    )


    # --------------------------------------
    # GREEN SPROUT RANGE
    # --------------------------------------
    #
    # We use a relatively wide green range
    # to increase sprout sensitivity.
    #
    # Hue in OpenCV HSV:
    # 0 - 179
    #
    # Green approximately:
    # 35 - 95
    # --------------------------------------

    lower_green = np.array([
        30,
        25,
        25
    ])

    upper_green = np.array([
        100,
        255,
        255
    ])


    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )


    # --------------------------------------
    # REMOVE SMALL NOISE
    # --------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )


    green_mask = cv2.morphologyEx(

        green_mask,

        cv2.MORPH_OPEN,

        kernel,

        iterations=1

    )


    green_mask = cv2.morphologyEx(

        green_mask,

        cv2.MORPH_DILATE,

        kernel,

        iterations=1

    )


    # --------------------------------------
    # CALCULATE GREEN PIXEL RATIO
    # --------------------------------------

    total_pixels = (
        image_roi.shape[0]
        *
        image_roi.shape[1]
    )


    green_pixels = cv2.countNonZero(
        green_mask
    )


    green_ratio = (
        green_pixels
        /
        max(total_pixels, 1)
    )


    # --------------------------------------
    # FIND GREEN COMPONENTS
    # --------------------------------------

    contours, _ = cv2.findContours(

        green_mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )


    largest_area = 0

    elongated_component_found = False


    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        if area < 5:

            continue


        largest_area = max(
            largest_area,
            area
        )


        x, y, w, h = cv2.boundingRect(
            contour
        )


        # Sprouts often form elongated
        # structures.

        if w > 0 and h > 0:

            aspect_ratio = max(
                w / h,
                h / w
            )


            if aspect_ratio > 1.3:

                elongated_component_found = True


    # --------------------------------------
    # CALCULATE SPROUT SCORE
    # --------------------------------------

    # Convert green ratio to score.

    ratio_score = min(

        green_ratio
        /
        max(
            SPROUT_GREEN_RATIO_THRESHOLD,
            0.0001
        ),

        1.0

    )


    # Elongated green component gives
    # additional confidence.

    shape_bonus = 0.25 if elongated_component_found else 0.0


    sprout_confidence = min(

        ratio_score * 0.75
        +
        shape_bonus,

        1.0

    )


    # --------------------------------------
    # FINAL DECISION
    # --------------------------------------

    is_sprouted = (

        green_ratio
        >=
        SPROUT_GREEN_RATIO_THRESHOLD

    )


    return (

        is_sprouted,

        round(
            sprout_confidence,
            3
        ),

        round(
            green_ratio,
            5
        )

    )


# ==========================================
# CALCULATE IMAGE FEATURES
# ==========================================

def calculate_onion_features(image_roi):

    """
    Calculate simple visual measurements.
    """

    if image_roi is None:

        return {
            "Area (px²)": 0,
            "Width (px)": 0,
            "Height (px)": 0,
            "Aspect Ratio": 0,
            "Circularity": 0,
            "Brightness": 0
        }


    if image_roi.size == 0:

        return {
            "Area (px²)": 0,
            "Width (px)": 0,
            "Height (px)": 0,
            "Aspect Ratio": 0,
            "Circularity": 0,
            "Brightness": 0
        }


    height, width = image_roi.shape[:2]


    # --------------------------------------
    # AREA
    # --------------------------------------

    area = width * height


    # --------------------------------------
    # ASPECT RATIO
    # --------------------------------------

    if height > 0:

        aspect_ratio = (
            width
            /
            height
        )

    else:

        aspect_ratio = 0


    # --------------------------------------
    # CIRCULARITY
    # --------------------------------------

    circularity = 0


    try:

        gray = cv2.cvtColor(

            image_roi,

            cv2.COLOR_BGR2GRAY

        )


        blurred = cv2.GaussianBlur(

            gray,

            (5, 5),

            0

        )


        _, threshold = cv2.threshold(

            blurred,

            0,

            255,

            cv2.THRESH_BINARY
            +
            cv2.THRESH_OTSU

        )


        contours, _ = cv2.findContours(

            threshold,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE

        )


        if contours:

            largest_contour = max(

                contours,

                key=cv2.contourArea

            )


            contour_area = cv2.contourArea(

                largest_contour

            )


            perimeter = cv2.arcLength(

                largest_contour,

                True

            )


            if perimeter > 0:

                circularity = (

                    4
                    *
                    np.pi
                    *
                    contour_area

                    /
                    (
                        perimeter
                        *
                        perimeter
                    )

                )

    except Exception:

        circularity = 0


    # --------------------------------------
    # BRIGHTNESS
    # --------------------------------------

    gray = cv2.cvtColor(

        image_roi,

        cv2.COLOR_BGR2GRAY

    )


    brightness = float(

        np.mean(
            gray
        )

    )


    return {

        "Area (px²)": round(
            area,
            1
        ),

        "Width (px)": int(
            width
        ),

        "Height (px)": int(
            height
        ),

        "Aspect Ratio": round(
            aspect_ratio,
            3
        ),

        "Circularity": round(
            circularity,
            3
        ),

        "Brightness": round(
            brightness,
            2
        )

    }


# ==========================================
# MAIN ANALYSIS FUNCTION
# ==========================================

def analyze_onions(image_path):


    # ======================================
    # LOAD IMAGE
    # ======================================

    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        raise ValueError(

            f"Could not load image:\n"
            f"{image_path}"

        )


    original_image = image.copy()


    image_height, image_width = (
        image.shape[:2]
    )


    # ======================================
    # RUN ROBOFLOW INFERENCE
    # ======================================

    result = CLIENT.infer(

        image,

        model_id=MODEL_ID

    )


    predictions = result.get(

        "predictions",

        []

    )


    onion_results = []


    healthy_count = 0

    rotten_count = 0

    sprouted_count = 0


    # ======================================
    # PROCESS DETECTIONS
    # ======================================

    for index, prediction in enumerate(

        predictions,

        start=1

    ):


        # ----------------------------------
        # CLASS
        # ----------------------------------

        detected_class = str(

            prediction.get(

                "class",

                "unknown"

            )

        )


        # ----------------------------------
        # DETECTION CONFIDENCE
        # ----------------------------------

        detection_confidence = float(

            prediction.get(

                "confidence",

                0

            )

        )


        # ----------------------------------
        # ROBOTFLOW BOX FORMAT
        # ----------------------------------

        center_x = float(

            prediction.get(

                "x",

                0

            )

        )


        center_y = float(

            prediction.get(

                "y",

                0

            )

        )


        box_width = float(

            prediction.get(

                "width",

                0

            )

        )


        box_height = float(

            prediction.get(

                "height",

                0

            )

        )


        # ----------------------------------
        # CONVERT TO CORNER COORDINATES
        # ----------------------------------

        x1 = int(

            center_x
            -
            box_width / 2

        )


        y1 = int(

            center_y
            -
            box_height / 2

        )


        x2 = int(

            center_x
            +
            box_width / 2

        )


        y2 = int(

            center_y
            +
            box_height / 2

        )


        # ----------------------------------
        # KEEP INSIDE IMAGE
        # ----------------------------------

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )


        x2 = min(
            image_width,
            x2
        )

        y2 = min(
            image_height,
            y2
        )


        # ----------------------------------
        # GET ONION CROP
        # ----------------------------------

        onion_roi = original_image[

            y1:y2,

            x1:x2

        ]


        # ----------------------------------
        # BASE CONDITION
        # ----------------------------------

        condition = map_to_condition(

            detected_class

        )


        condition_confidence = (

            detection_confidence

        )


        # ==================================
        # SPROUT FALLBACK DETECTION
        # ==================================

        is_sprouted, sprout_confidence, green_ratio = (

            detect_sprout(

                onion_roi

            )

        )


        # ----------------------------------
        # IMPORTANT:
        #
        # If Roboflow says sprouted OR
        # OpenCV finds sprout evidence,
        # mark as Sprouted.
        # ----------------------------------

        detected_class_lower = (

            detected_class

            .lower()

            .strip()

        )


        if (

            detected_class_lower
            ==
            "sprouted"

        ):

            condition = "Sprouted"

            condition_confidence = (

                detection_confidence

            )


        elif (

            is_sprouted

            and

            sprout_confidence >= 0.45

        ):

            condition = "Sprouted"

            condition_confidence = (

                sprout_confidence

            )


        # ----------------------------------
        # FEATURES
        # ----------------------------------

        features = calculate_onion_features(

            onion_roi

        )


        # ----------------------------------
        # COUNT CONDITION
        # ----------------------------------

        if condition == "Healthy":

            healthy_count += 1


        elif condition == "Sprouted":

            sprouted_count += 1


        else:

            # Everything else unhealthy

            condition = "Rotten"

            rotten_count += 1


        # ----------------------------------
        # BOX COLOR
        # ----------------------------------

        color = get_color(

            condition

        )


        # ----------------------------------
        # DRAW BOX
        # ----------------------------------

        cv2.rectangle(

            image,

            (x1, y1),

            (x2, y2),

            color,

            3

        )


        # ----------------------------------
        # DRAW LABEL
        # ----------------------------------

        label = (

            f"{condition}: "

            f"{condition_confidence:.2f}"

        )


        cv2.putText(

            image,

            label,

            (

                x1,

                max(
                    25,
                    y1 - 10
                )

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            color,

            2

        )


        # ==================================
        # STORE RESULT
        # ==================================

        onion_results.append({

            "Onion":

                index,


            "Detection Confidence":

                round(
                    detection_confidence,
                    3
                ),


            "Condition":

                condition,


            "Condition Confidence":

                round(
                    condition_confidence,
                    3
                ),


            "Detected Class":

                detected_class,


            "Sprout Evidence":

                round(
                    sprout_confidence,
                    3
                ),


            "Green Ratio":

                green_ratio,


            "Area (px²)":

                features[
                    "Area (px²)"
                ],


            "Width (px)":

                features[
                    "Width (px)"
                ],


            "Height (px)":

                features[
                    "Height (px)"
                ],


            "Aspect Ratio":

                features[
                    "Aspect Ratio"
                ],


            "Circularity":

                features[
                    "Circularity"
                ],


            "Brightness":

                features[
                    "Brightness"
                ],


            "Bounding Box": {

                "x1": x1,

                "y1": y1,

                "x2": x2,

                "y2": y2

            }

        })


    # ======================================
    # TOTAL ONIONS
    # ======================================

    total_onions = len(

        onion_results

    )


    # ======================================
    # RETURN RESULTS
    # ======================================

    return {

        "total_onions":

            total_onions,


        "healthy":

            healthy_count,


        "rotten":

            rotten_count,


        "sprouted":

            sprouted_count,


        "manual_review":

            0,


        "results":

            onion_results,


        "output_image":

            image

    }
