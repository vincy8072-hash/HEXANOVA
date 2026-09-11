def analyse_anomaly(results, confidence_threshold=0.50):

    boxes = results[0].boxes

    # No object detected
    if boxes is None or len(boxes) == 0:
        return {
            "status": "Potential Anomaly",
            "score": 1.0,
            "message": "No known trained object was confidently detected."
        }

    # Get highest confidence detection
    highest_confidence = max(
        float(box.conf[0])
        for box in boxes
    )

    # Known object confidently detected
    if highest_confidence >= confidence_threshold:
        return {
            "status": "Known Object",
            "score": 1 - highest_confidence,
            "message": "The detected pattern matches a known trained class."
        }

    # Detection exists but confidence is low
    else:
        return {
            "status": "Potential Anomaly",
            "score": 1 - highest_confidence,
            "message": "The detected pattern does not confidently match a known class."
        }