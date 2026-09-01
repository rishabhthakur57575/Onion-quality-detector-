import os
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from roboflow_grading import analyze_onions


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Onion Quality Assessment System",
    page_icon="🧅",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .title-text {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle-text {
        font-size: 18px;
        text-align: center;
        color: #9ca3af;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        background-color: #1c2330;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def get_output_image(result):

    """
    Returns the output image in a format Streamlit can display.

    Supports:
    - File path
    - PIL Image
    - NumPy/OpenCV image
    """

    output_image = result.get("output_image")

    if output_image is None:
        return None

    # If output is a file path
    if isinstance(output_image, (str, Path)):

        output_path = Path(output_image)

        if output_path.exists():
            return Image.open(output_path)

        return None

    # Otherwise return image object directly
    return output_image


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("⚙️ System Information")

    st.markdown("### Detection Model")

    st.success("Roboflow Object Detection")

    st.markdown("---")

    st.markdown("### Quality Categories")

    st.markdown("🟢 **Healthy**")

    st.markdown("🔴 **Rotten**")

    st.markdown("🌱 **Sprouted**")

    st.markdown("---")

    st.markdown("### Workflow")

    st.markdown(
        """
        **1.** Upload Image

        **2.** Run AI Detection

        **3.** Assess Onion Quality

        **4.** Display Results
        """
    )

    st.markdown("---")

    st.caption(
        "AI-based Onion Detection and Quality Assessment"
    )


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="title-text">🧅 Onion Quality Assessment System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-text">'
    'Detect onions and classify them as Healthy, Rotten or Sprouted'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# FILE UPLOADER
# ==========================================================

uploaded_file = st.file_uploader(
    "📤 Upload Onion Image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================================
# PROCESS IMAGE
# ==========================================================

if uploaded_file is not None:

    # ------------------------------------------------------
    # SAVE UPLOADED FILE TEMPORARILY
    # ------------------------------------------------------

    suffix = Path(uploaded_file.name).suffix

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_file.write(
        uploaded_file.getbuffer()
    )

    temp_file.close()

    temp_image_path = temp_file.name


    # ------------------------------------------------------
    # LOAD ORIGINAL IMAGE
    # ------------------------------------------------------

    original_image = Image.open(
        uploaded_file
    )


    # ======================================================
    # DISPLAY ORIGINAL IMAGE + ANALYSIS AREA
    # ======================================================

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("📷 Original Image")

        st.image(
            original_image,
            use_container_width=True
        )


    with col2:

        st.subheader("🤖 AI Analysis Result")

        st.info(
            "Click the Analyze button to run the AI model."
        )


    # ======================================================
    # ANALYZE BUTTON
    # ======================================================

    analyze_button = st.button(
        "🔍 Analyze Onions",
        type="primary",
        use_container_width=True
    )


    # ======================================================
    # RUN ANALYSIS
    # ======================================================

    if analyze_button:

        try:

            with st.spinner(
                "Running Roboflow AI detection..."
            ):

                result = analyze_onions(
                    temp_image_path
                )


            # ==================================================
            # EXTRACT RESULTS
            # ==================================================

            total_onions = result.get(
                "total_onions",
                0
            )

            healthy_count = result.get(
                "healthy",
                0
            )

            rotten_count = result.get(
                "rotten",
                0
            )

            sprouted_count = result.get(
                "sprouted",
                0
            )

            onion_results = result.get(
                "results",
                []
            )


            # ==================================================
            # DISPLAY SUCCESS MESSAGE
            # ==================================================

            st.success(
                "AI analysis completed successfully!"
            )


            # ==================================================
            # RESULTS IMAGE
            # ==================================================

            st.markdown("---")

            col1, col2 = st.columns(2)


            with col1:

                st.subheader("📷 Original Image")

                st.image(
                    original_image,
                    use_container_width=True
                )

            with col2:

                st.subheader("🤖 AI Analysis Result")

                output_image = get_output_image(result)

                if output_image is not None:

                    st.image(
                        output_image,
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "Annotated output image could not be displayed."
                    )


            # ==================================================
            # SUMMARY METRICS
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📊 Detection Summary"
            )


            metric1, metric2, metric3, metric4 = st.columns(
                4
            )


            with metric1:

                st.metric(
                    "🧅 Total Onions",
                    total_onions
                )


            with metric2:

                st.metric(
                    "🟢 Healthy",
                    healthy_count
                )


            with metric3:

                st.metric(
                    "🔴 Rotten",
                    rotten_count
                )


            with metric4:

                st.metric(
                    "🌱 Sprouted",
                    sprouted_count
                )


            # ==================================================
            # INDIVIDUAL ONION RESULTS TABLE
            # ==================================================

            st.markdown("---")

            st.subheader(
                "🧅 Individual Onion Results"
            )


            table_data = []


            for index, onion in enumerate(
                onion_results,
                start=1
            ):

                # ----------------------------------------------
                # ONION NUMBER
                # ----------------------------------------------

                onion_number = onion.get(
                    "Onion Number",
                    onion.get(
                        "onion_number",
                        index
                    )
                )


                # ----------------------------------------------
                # DETECTION CONFIDENCE
                # ----------------------------------------------

                detection_confidence = onion.get(
                    "Detection Confidence",
                    onion.get(
                        "Confidence",
                        onion.get(
                            "confidence",
                            0
                        )
                    )
                )


                # ----------------------------------------------
                # CONDITION
                # ----------------------------------------------

                condition = onion.get(
                    "Condition",
                    onion.get(
                        "condition",
                        "Rotten"
                    )
                )


                # Safety replacement:
                # Manual Review and Unhealthy should never
                # appear on dashboard.

                if str(condition).lower() in [
                    "manual review",
                    "manual_review",
                    "unhealthy",
                    "unknown"
                ]:
                    condition = "Rotten"


                # ----------------------------------------------
                # CONDITION CONFIDENCE
                # ----------------------------------------------

                condition_confidence = onion.get(
                    "Condition Confidence",
                    onion.get(
                        "condition_confidence",
                        detection_confidence
                    )
                )


                # ----------------------------------------------
                # IMAGE ANALYSIS VALUES
                # ----------------------------------------------

                area = onion.get(
                    "Area (px²)",
                    onion.get(
                        "Area",
                        onion.get(
                            "area",
                            0
                        )
                    )
                )


                width = onion.get(
                    "Width (px)",
                    onion.get(
                        "Width",
                        onion.get(
                            "width",
                            0
                        )
                    )
                )


                height = onion.get(
                    "Height (px)",
                    onion.get(
                        "Height",
                        onion.get(
                            "height",
                            0
                        )
                    )
                )


                aspect_ratio = onion.get(
                    "Aspect Ratio",
                    onion.get(
                        "aspect_ratio",
                        0
                    )
                )


                circularity = onion.get(
                    "Circularity",
                    onion.get(
                        "circularity",
                        0
                    )
                )


                brightness = onion.get(
                    "Brightness",
                    onion.get(
                        "brightness",
                        0
                    )
                )


                # ----------------------------------------------
                # ADD ROW
                # ----------------------------------------------

                table_data.append(
                    {
                        "Onion": onion_number,

                        "Detection Confidence":
                            round(
                                float(
                                    detection_confidence
                                ),
                                3
                            ),

                        "Condition":
                            condition,

                        "Condition Confidence":
                            round(
                                float(
                                    condition_confidence
                                ),
                                3
                            ),

                        "Area (px²)":
                            round(
                                float(area),
                                2
                            ),

                        "Width (px)":
                            round(
                                float(width),
                                2
                            ),

                        "Height (px)":
                            round(
                                float(height),
                                2
                            ),

                        "Aspect Ratio":
                            round(
                                float(
                                    aspect_ratio
                                ),
                                3
                            ),

                        "Circularity":
                            round(
                                float(
                                    circularity
                                ),
                                3
                            ),

                        "Brightness":
                            round(
                                float(
                                    brightness
                                ),
                                2
                            )
                    }
                )


            # ==================================================
            # DISPLAY TABLE
            # ==================================================

            if table_data:

                df = pd.DataFrame(
                    table_data
                )


                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )


                # ==============================================
                # DOWNLOAD CSV
                # ==============================================

                csv_data = df.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                )


                st.download_button(
                    label="⬇️ Download CSV Report",
                    data=csv_data,
                    file_name="onion_quality_report.csv",
                    mime="text/csv"
                )


                # ==============================================
                # DOWNLOAD JSON
                # ==============================================

                json_data = json.dumps(
                    table_data,
                    indent=4
                )


                st.download_button(
                    label="⬇️ Download JSON Report",
                    data=json_data,
                    file_name="onion_quality_report.json",
                    mime="application/json"
                )


            else:

                st.warning(
                    "No onion detections were returned."
                )


            # ==================================================
            # QUALITY SUMMARY
            # ==================================================

            st.markdown("---")

            st.subheader(
                "📈 Quality Assessment"
            )


            if total_onions > 0:

                healthy_percentage = (
                    healthy_count
                    / total_onions
                ) * 100


                rotten_percentage = (
                    rotten_count
                    / total_onions
                ) * 100


                sprouted_percentage = (
                    sprouted_count
                    / total_onions
                ) * 100


                chart_data = pd.DataFrame(
                    {
                        "Category": [
                            "Healthy",
                            "Rotten",
                            "Sprouted"
                        ],

                        "Count": [
                            healthy_count,
                            rotten_count,
                            sprouted_count
                        ]
                    }
                )


                st.bar_chart(
                    chart_data.set_index(
                        "Category"
                    )
                )


                percentage_col1, percentage_col2, percentage_col3 = (
                    st.columns(3)
                )


                with percentage_col1:

                    st.info(
                        f"🟢 Healthy: "
                        f"{healthy_percentage:.1f}%"
                    )


                with percentage_col2:

                    st.error(
                        f"🔴 Rotten: "
                        f"{rotten_percentage:.1f}%"
                    )


                with percentage_col3:

                    st.warning(
                        f"🌱 Sprouted: "
                        f"{sprouted_percentage:.1f}%"
                    )


            else:

                st.warning(
                    "No onions detected in this image."
                )


        except Exception as e:

            st.error(
                "An error occurred during AI analysis."
            )

            st.exception(
                e
            )


        finally:

            # Remove temporary image file

            if os.path.exists(
                temp_image_path
            ):

                try:

                    os.remove(
                        temp_image_path
                    )

                except Exception:

                    pass


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "🧅 Onion Quality Assessment System | "
    "AI-Powered Detection using Roboflow"
)
