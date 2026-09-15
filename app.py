import streamlit as st
from PIL import Image
import numpy as np
import cv2
import json
from datetime import datetime
from ultralytics import YOLO
from anomaly import analyse_anomaly


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HEXANOVA | Sonar Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "best.pt"
)

CONFIDENCE_THRESHOLD = 0.30


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

if "report" not in st.session_state:
    st.session_state.report = None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# PAGE LIST
# ============================================================

PAGES = [
    "Dashboard",
    "Upload & Analyse",
    "AI Detection + Anomaly Analysis",
    "Reports",
    "About System"
]


# ============================================================
# NAVIGATION FUNCTION
# ============================================================

def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def run_analysis(uploaded_file):

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_array = np.array(image)


    # --------------------------------------------------------
    # SONAR PRE-PROCESSING
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    filtered = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        filtered
    )

    processed_image = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2RGB
    )


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model(
        processed_image,
        conf=0.15
    )


    # --------------------------------------------------------
    # ANOMALY ANALYSIS
    # --------------------------------------------------------

    anomaly_result = analyse_anomaly(
        results,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        image_array=processed_image
    )


    # --------------------------------------------------------
    # DETECTION INFORMATION
    # --------------------------------------------------------

    boxes = results[0].boxes

    detection_data = []

    if boxes is not None and len(boxes) > 0:

        for i, box in enumerate(boxes):

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            class_name = model.names[
                class_id
            ]

            detection_data.append(
                {
                    "object": i + 1,
                    "class": class_name,
                    "confidence": round(
                        confidence,
                        4
                    )
                }
            )


    # --------------------------------------------------------
    # ANALYSIS RESULT
    # --------------------------------------------------------

    return {

        "image": image,

        "image_array": image_array,

        "processed_image": processed_image,

        "results": results,

        "boxes": boxes,

        "detections": detection_data,

        "anomaly": anomaly_result,

        "image_name": uploaded_file.name,

        "image_type": uploaded_file.type,

        "analysis_time":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

    }


# ============================================================
# HEADER
# ============================================================

st.title("🌊 HEXANOVA")

st.caption(
    "AI-Powered Underwater Marine Debris & "
    "Anomaly Detection using Side-Scan Sonar Imagery"
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.header("🌊 HEXANOVA")

    st.caption(
        "Underwater Sonar Intelligence Platform"
    )

    st.divider()

    selected_page = st.radio(
        "NAVIGATION",
        PAGES,
        index=PAGES.index(
            st.session_state.page
        )
    )

    if selected_page != st.session_state.page:

        st.session_state.page = selected_page

        st.rerun()

    st.divider()

    st.subheader("SYSTEM STATUS")

    st.success(
        "🟢 AI Model Online"
    )

    st.caption(
        "YOLOv8n • Side-Scan Sonar"
    )

    st.divider()

    st.info(
        "Current validated AI class:\n\n"
        "**Shipwreck**"
    )

    st.caption(
        "Anomaly analysis is a review layer. "
        "It does not automatically confirm marine debris."
    )


# ============================================================
# WORKFLOW INDICATOR
# ============================================================

current_index = PAGES.index(
    st.session_state.page
)

st.progress(
    (current_index + 1) / len(PAGES),
    text=(
        f"Step {current_index + 1} of "
        f"{len(PAGES)} — "
        f"{st.session_state.page}"
    )
)


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.header(
        "Mission Dashboard"
    )

    st.write(
        "Automated analysis overview for "
        "underwater Side-Scan Sonar imagery."
    )


    # --------------------------------------------------------
    # SYSTEM OVERVIEW
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "AI Engine",
            "YOLOv8"
        )

    with col2:

        st.metric(
            "Sonar Mode",
            "SSS"
        )

    with col3:

        st.metric(
            "Known AI Class",
            "Shipwreck"
        )

    with col4:

        if st.session_state.analysis_done:

            st.metric(
                "Status",
                "ANALYSED"
            )

        else:

            st.metric(
                "Status",
                "READY"
            )


    st.divider()


    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.subheader(
        "🔄 Analysis Workflow"
    )

    workflow_col1, workflow_col2, workflow_col3, workflow_col4, workflow_col5 = (
        st.columns(5)
    )

    with workflow_col1:

        st.info(
            "**01 — Upload**\n\n"
            "Upload sonar image"
        )

    with workflow_col2:

        st.info(
            "**02 — Pre-process**\n\n"
            "Noise + contrast"
        )

    with workflow_col3:

        st.info(
            "**03 — Detect**\n\n"
            "YOLO AI"
        )

    with workflow_col4:

        st.info(
            "**04 — Analyse**\n\n"
            "Anomaly layer"
        )

    with workflow_col5:

        st.info(
            "**05 — Report**\n\n"
            "Survey report"
        )


    st.divider()


    # --------------------------------------------------------
    # CURRENT ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "📊 Latest Analysis"
    )

    if st.session_state.analysis_done:

        data = st.session_state.analysis_data

        detections = data[
            "detections"
        ]

        anomaly = data[
            "anomaly"
        ]

        c1, c2, c3 = st.columns(3)

        with c1:

            if detections:

                best_detection = max(
                    detections,
                    key=lambda x:
                    x["confidence"]
                )

                st.success(
                    f"Detected: "
                    f"{best_detection['class']}"
                )

                st.caption(
                    f"Confidence: "
                    f"{best_detection['confidence']:.1%}"
                )

            else:

                st.info(
                    "No known object detected"
                )

        with c2:

            st.metric(
                "Anomaly Score",
                f"{anomaly['score'] * 100:.1f}%"
            )

        with c3:

            st.metric(
                "Suspicious Regions",
                len(
                    anomaly[
                        "anomaly_regions"
                    ]
                )
            )

    else:

        st.info(
            "No analysis has been performed yet. "
            "Start by uploading a Side-Scan Sonar image."
        )


    st.divider()


    # --------------------------------------------------------
    # START BUTTON
    # --------------------------------------------------------

    if st.button(
        "Next → Upload & Analyse",
        type="primary",
        use_container_width=True
    ):

        go_to(
            "Upload & Analyse"
        )


# ============================================================
# UPLOAD & ANALYSE
# ============================================================

elif st.session_state.page == "Upload & Analyse":

    st.header(
        "Upload & Analyse"
    )

    st.write(
        "Upload a Side-Scan Sonar image and "
        "run the HEXANOVA analysis pipeline."
    )


    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Side-Scan Sonar Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # IMAGE + INFORMATION
        # ----------------------------------------------------

        image_col, info_col = st.columns(
            [1.5, 1]
        )

        with image_col:

            st.subheader(
                "📷 Raw Sonar Image"
            )

            st.image(
                image,
                caption="Original Side-Scan Sonar Image",
                use_container_width=True
            )

        with info_col:

            st.subheader(
                "Image Information"
            )

            st.metric(
                "Width",
                f"{image.width} px"
            )

            st.metric(
                "Height",
                f"{image.height} px"
            )

            st.metric(
                "Format",
                uploaded_file.type
            )

        st.divider()


        # ----------------------------------------------------
        # ANALYSE BUTTON
        # ----------------------------------------------------

        if st.button(
            "🚀 RUN HEXANOVA ANALYSIS",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "🤖 Processing sonar imagery..."
            ):

                analysis = run_analysis(
                    uploaded_file
                )

                st.session_state.analysis_data = (
                    analysis
                )

                st.session_state.analysis_done = (
                    True
                )

                st.session_state.report = None


            st.success(
                "✅ Analysis completed successfully."
            )


            if st.button(
                "Next → AI Detection + Anomaly Analysis",
                type="primary",
                use_container_width=True
            ):

                go_to(
                    "AI Detection + Anomaly Analysis"
                )


        elif st.session_state.analysis_done:

            st.info(
                "A previous analysis is available."
            )

            if st.button(
                "Next → AI Detection + Anomaly Analysis",
                type="primary",
                use_container_width=True
            ):

                go_to(
                    "AI Detection + Anomaly Analysis"
                )

    else:

        st.info(
            "👆 Upload a Side-Scan Sonar image to begin."
        )


    st.divider()


    if st.button(
        "← Previous: Dashboard",
        use_container_width=True
    ):

        go_to(
            "Dashboard"
        )


# ============================================================
# AI DETECTION + ANOMALY ANALYSIS
# ============================================================

elif (
    st.session_state.page
    == "AI Detection + Anomaly Analysis"
):

    st.header(
        "AI Detection + Anomaly Analysis"
    )

    st.write(
        "Complete sonar interpretation from "
        "raw image to AI detection and anomaly review."
    )


    if not st.session_state.analysis_done:

        st.warning(
            "No analysis is available yet. "
            "Please upload a sonar image first."
        )

        if st.button(
            "← Go to Upload & Analyse",
            type="primary",
            use_container_width=True
        ):

            go_to(
                "Upload & Analyse"
            )

    else:

        data = st.session_state.analysis_data

        image = data["image"]

        processed_image = data[
            "processed_image"
        ]

        results = data[
            "results"
        ]

        detections = data[
            "detections"
        ]

        anomaly = data[
            "anomaly"
        ]


        # ====================================================
        # STEP 1 — RAW IMAGE
        # ====================================================

        st.subheader(
            "Step 1 — Raw Sonar Image"
        )

        st.image(
            image,
            caption="Original Side-Scan Sonar Image",
            use_container_width=True
        )


        st.divider()


        # ====================================================
        # STEP 2 — PREPROCESSING
        # ====================================================

        st.subheader(
            "Step 2 — Sonar Pre-processing"
        )

        st.write(
            "Noise reduction and contrast enhancement "
            "are applied before AI analysis."
        )

        st.image(
            processed_image,
            caption="Noise Reduction + Contrast Enhancement",
            use_container_width=True
        )


        st.divider()


        # ====================================================
        # STEP 3 — AI DETECTION
        # ====================================================

        st.subheader(
            "Step 3 — AI Detection"
        )

        annotated_image = results[
            0
        ].plot()

        st.image(
            annotated_image,
            caption="AI Detection Result",
            use_container_width=True
        )


        # ----------------------------------------------------
        # DETECTION RESULTS
        # ----------------------------------------------------

        if detections:

            confirmed = [
                d for d in detections
                if d["confidence"]
                >= CONFIDENCE_THRESHOLD
            ]

            low_confidence = [
                d for d in detections
                if d["confidence"]
                < CONFIDENCE_THRESHOLD
            ]


            if confirmed:

                st.success(
                    f"🟢 {len(confirmed)} "
                    f"known object(s) confirmed."
                )

            else:

                st.info(
                    "No known object was confidently "
                    "confirmed."
                )


            if low_confidence:

                st.warning(
                    f"🟠 {len(low_confidence)} "
                    f"low-confidence prediction(s) "
                    f"require review."
                )


            for detection in detections:

                confidence = detection[
                    "confidence"
                ]

                if confidence >= CONFIDENCE_THRESHOLD:

                    st.success(
                        f"**{detection['class']}** — "
                        f"Confidence: "
                        f"{confidence:.1%}"
                    )

                else:

                    st.warning(
                        f"**{detection['class']}** — "
                        f"Confidence: "
                        f"{confidence:.1%} "
                        f"(Low Confidence)"
                    )

        else:

            st.info(
                "No known trained object was detected."
            )


        # ----------------------------------------------------
        # CONFIDENCE SCORE
        # ----------------------------------------------------

        st.markdown(
            "### Confidence Score"
        )

        st.write(
            "It shows how strongly the AI model "
            "matches the detected region to its "
            "trained object class."
        )

        st.caption(
            "Validation basis: the value is the YOLO "
            "model's detection confidence. In this "
            "prototype, 30% is used as the confirmation "
            "threshold; below that, the result is treated "
            "as low confidence."
        )


        if detections:

            best_detection = max(
                detections,
                key=lambda x:
                x["confidence"]
            )

            st.metric(
                "Best Confidence Score",
                f"{best_detection['confidence'] * 100:.1f}%"
            )


        st.divider()


        # ====================================================
        # STEP 4 — ANOMALY ANALYSIS
        # ====================================================

        st.subheader(
            "Step 4 — Anomaly Analysis"
        )


        score = float(
            anomaly.get(
                "score",
                0
            )
        )

        status = anomaly.get(
            "status",
            "Unknown"
        )

        regions = anomaly.get(
            "anomaly_regions",
            []
        )

        message = anomaly.get(
            "message",
            ""
        )

        explanation = anomaly.get(
            "explanation",
            ""
        )


        # ----------------------------------------------------
        # SCORE METRICS
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Anomaly Score",
                f"{score * 100:.1f}%"
            )

        with c2:

            st.metric(
                "Suspicious Regions",
                len(regions)
            )

        with c3:

            st.metric(
                "AI Targets",
                len(detections)
            )


        # ----------------------------------------------------
        # ANOMALY SCORE EXPLANATION
        # ----------------------------------------------------

        st.markdown(
            "### Anomaly Score"
        )

        st.write(
            "It shows how strongly the image contains "
            "unusual localized sonar patterns that "
            "may need further inspection."
        )

        st.caption(
            "Validation basis: the prototype anomaly "
            "layer looks for localized high-intensity "
            "regions after sonar preprocessing. The score "
            "is a prototype heuristic, not a calibrated "
            "probability of marine debris."
        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if status == "Known Object":

            st.success(
                "🟢 Known Object — "
                "No Unknown Anomaly Flag"
            )

        elif status == "Low Confidence Detection":

            st.warning(
                "🟠 Low Confidence Detection — "
                "Needs Review"
            )

        elif status == "Potential Anomaly":

            st.warning(
                "🔶 Potential Anomaly Detected — "
                "Needs Inspection"
            )

        elif status == "Normal / No Strong Anomaly":

            st.success(
                "🟢 Normal / No Strong Anomaly"
            )

        else:

            st.info(
                status
            )


        st.write(
            message
        )

        st.caption(
            explanation
        )


        # ----------------------------------------------------
        # SUSPICIOUS REGIONS
        # ----------------------------------------------------

        if regions:

            st.subheader(
                "🟨 Suspicious Sonar Regions"
            )

            anomaly_visual = (
                processed_image.copy()
            )


            for index, region in enumerate(
                regions
            ):

                x = region["x"]

                y = region["y"]

                w = region["width"]

                h = region["height"]


                cv2.rectangle(
                    anomaly_visual,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 0),
                    3
                )


                cv2.putText(
                    anomaly_visual,
                    f"Potential Region {index + 1}",
                    (
                        x,
                        max(
                            20,
                            y - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )


            st.image(
                anomaly_visual,
                caption="Potential Anomaly Regions",
                use_container_width=True
            )

        else:

            st.info(
                "No suspicious sonar regions "
                "were identified."
            )


        st.divider()


        # ----------------------------------------------------
        # FINAL INTERPRETATION
        # ----------------------------------------------------

        st.subheader(
            "📌 Analysis Interpretation"
        )

        if status == "Known Object":

            st.success(
                "The AI detected a pattern matching "
                "the currently trained Shipwreck class. "
                "It is therefore treated as a known "
                "object rather than an unknown anomaly."
            )

        elif status == "Low Confidence Detection":

            st.warning(
                "The AI found some similarity to the "
                "known class, but the confidence is below "
                "the confirmation threshold."
            )

        elif status == "Potential Anomaly":

            st.warning(
                "No known object was confidently detected. "
                "Localized sonar patterns were flagged "
                "for further human inspection."
            )

        else:

            st.info(
                "No strong known-object or suspicious "
                "anomaly pattern was identified."
            )


        st.divider()


        # ----------------------------------------------------
        # NAVIGATION
        # ----------------------------------------------------

        previous_col, next_col = st.columns(2)

        with previous_col:

            if st.button(
                "← Previous: Upload & Analyse",
                use_container_width=True
            ):

                go_to(
                    "Upload & Analyse"
                )

        with next_col:

            if st.button(
                "Next → Reports",
                type="primary",
                use_container_width=True
            ):

                go_to(
                    "Reports"
                )


# ============================================================
# REPORTS
# ============================================================

elif st.session_state.page == "Reports":

    st.header(
        "Survey Report"
    )

    st.write(
        "Review the complete survey result before "
        "downloading the structured report."
    )


    if not st.session_state.analysis_done:

        st.warning(
            "No survey report is available yet."
        )

        if st.button(
            "← Go to Upload & Analyse",
            type="primary",
            use_container_width=True
        ):

            go_to(
                "Upload & Analyse"
            )

    else:

        data = st.session_state.analysis_data

        image = data[
            "image"
        ]

        processed_image = data[
            "processed_image"
        ]

        results = data[
            "results"
        ]

        detections = data[
            "detections"
        ]

        anomaly = data[
            "anomaly"
        ]

        # ----------------------------------------------------
        # REPORT HEADER
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🌊 HEXANOVA — Sonar Survey Report"
        )

        st.caption(
            "Automated Side-Scan Sonar Intelligence Report"
        )


        # ----------------------------------------------------
        # SURVEY IMAGE
        # ----------------------------------------------------

        st.subheader(
            "🖼️ Survey Image"
        )

        st.image(
            image,
            caption="Original Side-Scan Sonar Image",
            use_container_width=True
        )


        # ----------------------------------------------------
        # DETECTION IMAGE
        # ----------------------------------------------------

        st.subheader(
            "🎯 AI Detection Image"
        )

        annotated_image = results[
            0
        ].plot()

        st.image(
            annotated_image,
            caption="Annotated AI Detection Result",
            use_container_width=True
        )


        st.divider()


        # ----------------------------------------------------
        # SURVEY SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "📋 Survey Summary"
        )

        c1, c2, c3 = st.columns(3)


        with c1:

            st.write(
                f"**Image:** "
                f"{data['image_name']}"
            )

        with c2:

            st.write(
                f"**Resolution:** "
                f"{image.width} × "
                f"{image.height}"
            )

        with c3:

            st.write(
                f"**Analysis Time:** "
                f"{data['analysis_time']}"
            )


        st.divider()


        st.divider()


        # ----------------------------------------------------
        # AI DETECTION SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "🤖 AI Detection Summary"
        )


        if detections:

            for detection in detections:

                confidence = detection[
                    "confidence"
                ]

                if confidence >= CONFIDENCE_THRESHOLD:

                    st.success(
                        f"🟢 **{detection['class']}** — "
                        f"Confidence: "
                        f"{confidence:.1%}"
                    )

                else:

                    st.warning(
                        f"🟠 **{detection['class']}** — "
                        f"Low Confidence: "
                        f"{confidence:.1%}"
                    )

        else:

            st.info(
                "No known trained object was detected."
            )


        if detections:

            best_detection = max(
                detections,
                key=lambda x:
                x["confidence"]
            )

            st.write(
                f"**Highest Confidence Score:** "
                f"{best_detection['confidence']:.1%}"
            )

            st.caption(
                "It shows how strongly the AI model "
                "matches a detected region to its "
                "trained object class."
            )


        st.divider()


        # ----------------------------------------------------
        # PREPROCESSING SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "🛰️ Sonar Pre-processing"
        )

        st.write(
            "Noise reduction and contrast enhancement "
            "were applied before AI detection."
        )

        with st.expander(
            "View Pre-processed Sonar Image"
        ):

            st.image(
                processed_image,
                caption="Pre-processed Sonar Image",
                use_container_width=True
            )


        st.divider()


        # ----------------------------------------------------
        # ANOMALY SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "⚠️ Anomaly Analysis Summary"
        )

        anomaly_status = anomaly.get(
            "status",
            "Unknown"
        )

        anomaly_score = float(
            anomaly.get(
                "score",
                0
            )
        )

        suspicious_regions = anomaly.get(
            "anomaly_regions",
            []
        )

        explanation = anomaly.get(
            "explanation",
            ""
        )


        anomaly_col1, anomaly_col2 = st.columns(2)


        with anomaly_col1:

            st.metric(
                "Anomaly Status",
                anomaly_status
            )


        with anomaly_col2:

            st.metric(
                "Anomaly Score",
                f"{anomaly_score * 100:.1f}%"
            )


        st.write(
            f"**Suspicious Regions:** "
            f"{len(suspicious_regions)}"
        )

        st.write(
            explanation
        )

        st.caption(
            "It shows how strongly the image contains "
            "unusual localized sonar patterns that may "
            "need further inspection. It is not a "
            "confirmed marine-debris probability."
        )


        st.divider()


        # ----------------------------------------------------
        # FINAL SURVEY CONCLUSION
        # ----------------------------------------------------

        st.subheader(
            "📝 Final Survey Conclusion"
        )


        if anomaly_status == "Known Object":

            st.success(
                "Known object detected. The sonar pattern "
                "matches the currently trained Shipwreck class."
            )

        elif anomaly_status == "Potential Anomaly":

            st.warning(
                "Potential anomaly identified. The flagged "
                "region requires further human inspection."
            )

        elif anomaly_status == "Low Confidence Detection":

            st.warning(
                "A weak known-object pattern was identified, "
                "but the confidence is below the confirmation "
                "threshold."
            )

        else:

            st.info(
                "No strong known-object or suspicious "
                "anomaly pattern was identified."
            )


        st.divider()


        # ----------------------------------------------------
        # REPORT DATA
        # ----------------------------------------------------

        report = {

            "report_title":
                "HEXANOVA Sonar Survey Report",

            "system":
                "HEXANOVA",

            "analysis_time":
                data["analysis_time"],

            "image_name":
                data["image_name"],

            "image_size": {

                "width":
                    image.width,

                "height":
                    image.height
            },

            "processing": {

                "noise_reduction":
                    True,

                "contrast_enhancement":
                    True,

                "ai_model":
                    "YOLOv8",

                "known_class":
                    "Shipwreck"
            },

            "detected_objects":
                detections,

            "confidence_threshold":
                CONFIDENCE_THRESHOLD,

            "anomaly_analysis": {

                "status":
                    anomaly_status,

                "score":
                    round(
                        anomaly_score,
                        4
                    ),

                "suspicious_region_count":
                    len(
                        suspicious_regions
                    ),

                "explanation":
                    explanation
            },

            "suspicious_regions":
                suspicious_regions
        }


        st.session_state.report = report


        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        st.subheader(
            "⬇️ Download Survey Report"
        )

        st.caption(
            "The downloaded JSON contains the structured "
            "survey information shown in this report."
        )


        report_json = json.dumps(
            report,
            indent=4
        )


        st.download_button(
            label="⬇️ Download HEXANOVA Survey Report",
            data=report_json,
            file_name=(
                "hexanova_sonar_survey_report.json"
            ),
            mime="application/json",
            use_container_width=True
        )


    st.divider()


    previous_col, next_col = st.columns(2)


    with previous_col:

        if st.button(
            "← Previous: AI Detection + Anomaly Analysis",
            use_container_width=True
        ):

            go_to(
                "AI Detection + Anomaly Analysis"
            )


    with next_col:

        if st.button(
            "Next → About System",
            type="primary",
            use_container_width=True
        ):

            go_to(
                "About System"
            )


# ============================================================
# ABOUT SYSTEM
# ============================================================

elif st.session_state.page == "About System":

    st.header(
        "About HEXANOVA"
    )

    st.subheader(
        "🌊 AI-Powered Underwater Marine Debris & "
        "Anomaly Detection System"
    )


    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    st.subheader(
        "🎯 Objective"
    )

    st.write(
        "HEXANOVA is a prototype designed to reduce "
        "the manual effort involved in analysing "
        "Side-Scan Sonar imagery."
    )


    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.subheader(
        "🧠 AI Pipeline"
    )

    st.info(
        "Raw Sonar Image"
        "  →  "
        "Sonar Pre-processing"
        "  →  "
        "AI Detection"
        "  →  "
        "Anomaly Analysis"
        "  →  "
        "Survey Report"
    )


    # --------------------------------------------------------
    # CURRENT AI
    # --------------------------------------------------------

    st.subheader(
        "🔬 Current Validated AI Capability"
    )

    st.success(
        "The current trained YOLO model is validated "
        "for the Shipwreck class using the "
        "AI4Shipwrecks Side-Scan Sonar dataset."
    )


    st.write(
        "The architecture can be extended with "
        "additional genuinely annotated Side-Scan "
        "Sonar marine-debris classes."
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    st.subheader(
        "Confidence Score"
    )

    st.write(
        "It shows how strongly the YOLO model "
        "matches a detected sonar region to its "
        "trained object class."
    )

    st.write(
        f"Prototype confirmation threshold: "
        f"**{CONFIDENCE_THRESHOLD:.0%}**"
    )


    # --------------------------------------------------------
    # ANOMALY
    # --------------------------------------------------------

    st.subheader(
        "Anomaly Score"
    )

    st.write(
        "It shows how strongly localized sonar "
        "patterns differ from the surrounding image "
        "and may require further inspection."
    )

    st.caption(
        "The current anomaly score is a prototype "
        "heuristic and should not be interpreted as "
        "a calibrated probability."
    )


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    st.subheader(
        "📄 Survey Report"
    )

    st.write(
        "The report combines the analysed sonar image, "
        "AI detection result, confidence, anomaly result, "
        "and structured analysis data."
    )


    # --------------------------------------------------------
    # IMPORTANT LIMITATION
    # --------------------------------------------------------

    st.subheader(
        "⚠️ Important"
    )

    st.warning(
        "A region is not automatically classified as "
        "marine debris simply because the YOLO model "
        "does not detect a known object."
    )

    st.write(
        "Potential anomaly regions are flagged for "
        "further inspection and human validation."
    )


    # --------------------------------------------------------
    # FUTURE EXTENSIONS
    # --------------------------------------------------------

    st.subheader(
        "🚀 Future Extensions"
    )

    st.write(
        """
        • Multi-class marine debris detection

        • Real sonar metadata extraction

        • Automatic GPS geotagging

        • Physical dimension estimation

        • Improved anomaly detection

        • Object-level sonar positioning

        • Edge deployment
        """
    )


    st.divider()


    if st.button(
        "← Previous: Reports",
        use_container_width=True
    ):

        go_to(
            "Reports"
        )
