import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from anomaly import analyse_anomaly


# Page configuration
st.set_page_config(
    page_title="Marine Debris & Anomaly Detection",
    page_icon="🌊",
    layout="wide"
)


# Load our trained Marine Debris YOLO model
@st.cache_resource
def load_model():
    return YOLO(
        r"C:\Users\vincy\OneDrive\Desktop\marine_debris_ai\runs\detect\train-5\weights\best.pt"
    )


model = load_model()


# Dashboard title
st.title("🌊 HEXANOVA")

st.write(
    "Upload an underwater or sonar image to detect "
    "marine debris and potential anomalous objects."
)


# Upload image
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Convert image to NumPy array
    image_array = np.array(image)

    # Display uploaded image
    st.subheader("📤 Uploaded Image")

    st.image(
        image,
        caption="Original Uploaded Image",
        use_container_width=True
    )


    # Image information
    st.subheader("📊 Image Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Image Size:", image.size)

    with col2:
        st.write("Image Shape:", image_array.shape)


    # Analyse button
    if st.button("🔍 Analyse Image"):

        with st.spinner("🤖 Marine debris AI analysis in progress..."):

            # Run trained YOLO model
            results = model(
                image_array,
                conf=0.50
            )

            # Run anomaly analysis based on YOLO detection
            anomaly_result = analyse_anomaly(
                results,
                confidence_threshold=0.50
            )

            # Get annotated image
            annotated_image = results[0].plot()


        # AI Detection Result
        st.subheader("🤖 AI Detection Result")

        st.image(
            annotated_image,
            caption="Marine Debris Detection Result",
            use_container_width=True
        )


        # Anomaly Analysis
        st.subheader("⚠️ Anomaly Analysis")

        st.metric(
            "Anomaly Score",
            f"{anomaly_result['score']:.2f}"
        )

        if anomaly_result["status"] == "Known Object":

            st.success(
                "✅ Known Object Pattern Detected"
            )

        else:

            st.warning(
                "⚠️ Potential Anomaly Detected"
            )

        st.write(
            anomaly_result["message"]
        )


        # Get detected boxes
        boxes = results[0].boxes


        # Detection Analysis
        st.subheader("📋 Detection Analysis")

        if boxes is not None and len(boxes) > 0:

            st.success(
                f"✅ {len(boxes)} known object(s) detected!"
            )

            # Display each detected object
            for i, box in enumerate(boxes):

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                class_name = model.names[class_id]

                st.write(
                    f"**Object {i + 1}:** "
                    f"{class_name} | "
                    f"Confidence: {confidence:.2%}"
                )

                st.info(
                    f"Classification: The trained AI found "
                    f"visual patterns matching the "
                    f"'{class_name}' class."
                )

        else:

            st.warning(
                "⚠️ No known trained object was confidently detected."
            )


else:

    st.info(
        "👆 Upload an underwater or sonar image to start AI analysis."
    )