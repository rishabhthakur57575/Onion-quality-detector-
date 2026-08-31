import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import json
import tempfile

from onion_analysis import analyze_onion


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Onion Quality Inspector",
    page_icon="🧅",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
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


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        text-align: center;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧅 Onion Quality Inspector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-assisted onion detection and quality assessment'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    detection_confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    quality_confidence = st.slider(
        "Quality confidence",
        min_value=0.50,
        max_value=0.99,
        value=0.75,
        step=0.05
    )

    st.divider()

    st.write("### Models")

    st.write(
        "🔍 Onion Detector"
    )

    st.write(
        "🧠 Healthy / Unhealthy Classifier"
    )

    st.divider()

    st.caption(
        "Phase 1B Prototype"
    )


# ============================================================
# CHECK MODELS
# ============================================================

if not DETECTOR_PATH.exists():

    st.error(
        f"Detector model not found:\n{DETECTOR_PATH}"
    )

    st.stop()


if not QUALITY_MODEL_PATH.exists():

    st.error(
        f"Quality model not found:\n{QUALITY_MODEL_PATH}"
    )

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    detector = YOLO(
        str(DETECTOR_PATH)
    )

    quality_model = YOLO(
        str(QUALITY_MODEL_PATH)
    )

    return detector, quality_model


with st.spinner("Loading AI models..."):

    detector, quality_model = load_models()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📷 Upload Onion Batch'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an image containing onions",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# MAIN PROCESSING
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "Could not read the uploaded image."
        )

        st.stop()


    # --------------------------------------------------------
    # Display original image
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🔍 Inspection'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )


    # --------------------------------------------------------
    # Run YOLO detector
    # --------------------------------------------------------

    with st.spinner(
        "Detecting onions..."
    ):

        results = detector(
            image,
            conf=detection_confidence,
            verbose=False
        )

    result = results[0]


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    onion_data = []

    annotated_image = image.copy()


    # ========================================================
    # PROCESS EACH ONION
    # ========================================================

    for i, box in enumerate(
        result.boxes
    ):

        detection_conf = float(
            box.conf[0]
        )


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            image.shape[1],
            x2
        )

        y2 = min(
            image.shape[0],
            y2
        )


        # ----------------------------------------------------
        # Crop onion
        # ----------------------------------------------------

        crop = image[
            y1:y2,
            x1:x2
        ]


        if crop.size == 0:

            continue


        # ----------------------------------------------------
        # Quality classification
        # ----------------------------------------------------

        quality_results = quality_model(
            crop,
            verbose=False
        )

        quality_result = (
            quality_results[0]
        )


        class_id = int(
            quality_result.probs.top1
        )

        condition_conf = float(
            quality_result.probs.top1conf
        )

        condition = (
            quality_result.names[
                class_id
            ]
        )


        # ----------------------------------------------------
        # Manual review threshold
        # ----------------------------------------------------

        if condition_conf < quality_confidence:

            displayed_condition = (
                "Manual Review"
            )

        else:

            displayed_condition = (
                condition.title()
            )


        # ----------------------------------------------------
        # OpenCV analysis
        # ----------------------------------------------------

        analysis = analyze_onion(
            crop,
            debug=False
        )


        # ----------------------------------------------------
        # Store data
        # ----------------------------------------------------

        onion_info = {

            "Onion": i + 1,

            "Detection Confidence":
                round(
                    detection_conf,
                    3
                ),

            "Condition":
                displayed_condition,

            "Condition Confidence":
                round(
                    condition_conf,
                    3
                ),

            "X1": x1,
            "Y1": y1,
            "X2": x2,
            "Y2": y2
        }


        if analysis:

            onion_info.update({

                "Area (px²)":
                    analysis["area"],

                "Width (px)":
                    analysis["width"],

                "Height (px)":
                    analysis["height"],

                "Aspect Ratio":
                    analysis["aspect_ratio"],

                "Circularity":
                    analysis["circularity"],

                "Brightness":
                    analysis[
                        "mean_brightness"
                    ],

                "Analysis Method":
                    analysis["method"]
            })

        else:

            onion_info.update({

                "Area (px²)": None,

                "Width (px)": None,

                "Height (px)": None,

                "Aspect Ratio": None,

                "Circularity": None,

                "Brightness": None,

                "Analysis Method": "Failed"
            })


        onion_data.append(
            onion_info
        )


        # ====================================================
        # DRAW BOUNDING BOX
        # ====================================================

        if displayed_condition == "Healthy":

            box_color = (
                0,
                200,
                0
            )

        elif displayed_condition == "Unhealthy":

            box_color = (
                0,
                0,
                255
            )

        else:

            box_color = (
                0,
                165,
                255
            )


        cv2.rectangle(

            annotated_image,

            (x1, y1),

            (x2, y2),

            box_color,

            3
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"#{i + 1} "
            f"{displayed_condition} "
            f"{condition_conf:.2f}"
        )


        cv2.putText(

            annotated_image,

            label,

            (
                x1,
                max(
                    25,
                    y1 - 10
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            box_color,

            2
        )


    # ========================================================
    # BATCH STATISTICS
    # ========================================================

    total = len(
        onion_data
    )

    healthy_count = sum(

        1

        for onion in onion_data

        if onion["Condition"]
        == "Healthy"
    )

    unhealthy_count = sum(

        1

        for onion in onion_data

        if onion["Condition"]
        == "Unhealthy"
    )

    review_count = sum(

        1

        for onion in onion_data

        if onion["Condition"]
        == "Manual Review"
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


    # ========================================================
    # SHOW ANNOTATED IMAGE
    # ========================================================

    with col2:

        st.subheader(
            "AI Inspection Result"
        )

        st.image(

            cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            ),

            use_container_width=True
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Batch Summary'
        '</div>',
        unsafe_allow_html=True
    )


    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.metric(
            "Total Onions",
            total
        )


    with m2:

        st.metric(
            "Healthy",
            healthy_count,
            f"{healthy_percentage:.1f}%"
        )


    with m3:

        st.metric(
            "Unhealthy",
            unhealthy_count,
            f"{unhealthy_percentage:.1f}%"
        )


    with m4:

        st.metric(
            "Manual Review",
            review_count
        )


    # ========================================================
    # QUALITY BAR
    # ========================================================

    if total > 0:

        st.subheader(
            "Batch Condition"
        )

        st.progress(
            healthy_count / total
        )

        st.caption(
            f"{healthy_percentage:.1f}% "
            "of detected onions classified as healthy"
        )


    # ========================================================
    # INDIVIDUAL RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔬 Individual Onion Results'
        '</div>',
        unsafe_allow_html=True
    )


    if onion_data:

        df = pd.DataFrame(
            onion_data
        )

        display_columns = [

            "Onion",

            "Detection Confidence",

            "Condition",

            "Condition Confidence",

            "Area (px²)",

            "Width (px)",

            "Height (px)",

            "Aspect Ratio",

            "Circularity",

            "Brightness"
        ]

        st.dataframe(

            df[
                display_columns
            ],

            use_container_width=True,

            hide_index=True
        )


    # ========================================================
    # DOWNLOAD CSV
    # ========================================================

    if onion_data:

        csv_data = df.to_csv(
            index=False
        )

        st.download_button(

            label="⬇️ Download CSV Report",

            data=csv_data,

            file_name="onion_batch_report.csv",

            mime="text/csv"
        )


        # ----------------------------------------------------
        # JSON report
        # ----------------------------------------------------

        json_report = {

            "total_onions":
                total,

            "healthy":
                healthy_count,

            "unhealthy":
                unhealthy_count,

            "manual_review":
                review_count,

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

            "onions":
                onion_data
        }


        json_data = json.dumps(
            json_report,
            indent=4
        )


        st.download_button(

            label="⬇️ Download JSON Report",

            data=json_data,

            file_name="onion_batch_report.json",

            mime="application/json"
        )


    # ========================================================
    # PROJECT DISCLAIMER
    # ========================================================

    st.divider()

    st.info(
        "Prototype note: Healthy/Unhealthy classification "
        "is AI-assisted and should be manually reviewed when "
        "confidence is low. Physical size is reported in "
        "pixels until camera calibration is performed."
    )
