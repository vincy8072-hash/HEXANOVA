import cv2
import numpy as np


def analyse_anomaly(
    results,
    confidence_threshold=0.30,
    image_array=None
):
    """
    HEXANOVA anomaly analysis.

    Detection confidence and anomaly score
    are treated as two different measurements.
    """

    boxes = results[0].boxes

    # ========================================================
    # CASE 1 — KNOWN OBJECT DETECTED
    # ========================================================

    if boxes is not None and len(boxes) > 0:

        confidences = [
            float(box.conf[0])
            for box in boxes
        ]

        highest_confidence = max(confidences)
        object_count = len(boxes)

        # --------------------------------------------
        # CONFIDENT KNOWN OBJECT
        # --------------------------------------------

        if highest_confidence >= confidence_threshold:

            return {
                "status": "Known Object",
                "score": 0.10,
                "message": (
                    f"{object_count} known sonar object(s) "
                    "were confidently detected."
                ),
                "anomaly_regions": [],
                "explanation": (
                    "The AI identified a sonar pattern similar "
                    "to the known training class. Therefore, "
                    "the detected region is treated as a known "
                    "object rather than an unknown anomaly."
                ),
                "detection_confidence": highest_confidence
            }

        # --------------------------------------------
        # WEAK DETECTION
        # --------------------------------------------

        return {
            "status": "Low Confidence Detection",
            "score": 0.30,
            "message": (
                "A weak known-object pattern was detected, "
                "but the confidence is below the confirmation threshold."
            ),
            "anomaly_regions": [],
            "explanation": (
                "The sonar pattern shows some similarity to "
                "the known training class, but the AI confidence "
                "is not high enough to confirm the object."
            ),
            "detection_confidence": highest_confidence
        }

    # ========================================================
    # CASE 2 — NO YOLO DETECTION
    # ========================================================

    if image_array is None:

        return {
            "status": "Insufficient Evidence",
            "score": 0.20,
            "message": (
                "No known object was detected and image-based "
                "anomaly analysis could not be performed."
            ),
            "anomaly_regions": [],
            "explanation": (
                "There is insufficient sonar evidence to "
                "classify this image as a potential anomaly."
            ),
            "detection_confidence": 0.0
        }

    # ========================================================
    # SONAR IMAGE ANALYSIS
    # ========================================================

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ========================================================
    # CONTRAST ANALYSIS
    # ========================================================

    mean_intensity = float(
        np.mean(blurred)
    )

    std_intensity = float(
        np.std(blurred)
    )

    threshold_value = mean_intensity + (
        1.5 * std_intensity
    )

    threshold_value = max(
        0,
        min(
            255,
            threshold_value
        )
    )

    _, binary = cv2.threshold(
        blurred,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # ========================================================
    # NOISE REMOVAL
    # ========================================================

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ========================================================
    # FIND SUSPICIOUS REGIONS
    # ========================================================

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    anomaly_regions = []

    image_height, image_width = gray.shape

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = w * h

        if area < 150:
            continue

        if area > (
            image_width * image_height * 0.30
        ):
            continue

        anomaly_regions.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area": int(area)
        })

    # ========================================================
    # NORMAL SEABED
    # ========================================================

    region_count = len(
        anomaly_regions
    )

    if region_count == 0:

        return {
            "status": "Normal / No Strong Anomaly",
            "score": 0.15,
            "message": (
                "No known object was detected and no strong "
                "suspicious sonar region was identified."
            ),
            "anomaly_regions": [],
            "explanation": (
                "The sonar image does not contain a sufficiently "
                "strong localized region to be flagged as a "
                "potential artificial anomaly."
            ),
            "detection_confidence": 0.0
        }

    # ========================================================
    # POTENTIAL ANOMALY
    # ========================================================

    anomaly_score = min(
        0.85,
        0.25 + (region_count * 0.08)
    )

    return {
        "status": "Potential Anomaly",
        "score": anomaly_score,
        "message": (
            f"{region_count} suspicious sonar region(s) "
            "were identified for further inspection."
        ),
        "anomaly_regions": anomaly_regions,
        "explanation": (
            "No known trained object was confidently detected. "
            "However, localized high-intensity regions differ "
            "from the surrounding seabed and have therefore "
            "been flagged as potential anomalies. "
            "These regions are not confirmed marine debris."
        ),
        "detection_confidence": 0.0
    }
