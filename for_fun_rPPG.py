import cv2
import dlib
import time
import numpy as np
from collections import deque
from imutils.video import VideoStream

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Length of the rolling time window (seconds) for the green-channel signal
RPPG_WINDOW_SECONDS = 5.0

# Minimum number of samples before using the signal
MIN_SAMPLES_FOR_PULSE = 16

# Video source index (0 = default webcam)
CAMERA_INDEX = 0

# Colors
COLOR_FACE_BOX = (0, 255, 0)
COLOR_ROI_BOX = (255, 255, 0)
COLOR_TEXT = (255, 255, 255)


# ---------------------------------------------------------------------------
# Forehead ROI extraction
# ---------------------------------------------------------------------------

def extract_forehead_roi_from_face(frame, rect):
    """
    Define a forehead region-of-interest (ROI) using the face bounding box.

    The ROI is placed near the top-middle of the detected face rectangle.
    Returns (x, y, w, h) in pixel coordinates, or None if invalid.
    """
    h_frame, w_frame = frame.shape[:2]

    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_frame - 1, x2)
    y2 = min(h_frame - 1, y2)

    face_w = x2 - x1
    face_h = y2 - y1
    if face_w <= 0 or face_h <= 0:
        return None

    # Forehead area: top ~20% of the face, centered horizontally
    roi_w = int(face_w * 0.4)
    roi_h = int(face_h * 0.18)
    roi_x = int(x1 + face_w * 0.3)       # ~center horizontally
    roi_y = int(y1 + face_h * 0.10)      # near top

    # Clamp ROI to frame bounds
    roi_x = max(0, min(roi_x, w_frame - 1))
    roi_y = max(0, min(roi_y, h_frame - 1))
    roi_w = min(roi_w, w_frame - roi_x)
    roi_h = min(roi_h, h_frame - roi_y)

    if roi_w <= 0 or roi_h <= 0:
        return None

    return roi_x, roi_y, roi_w, roi_h


# ---------------------------------------------------------------------------
# Pulse intensity estimation and visualization
# ---------------------------------------------------------------------------

def get_pulse_intensity_simple(buffer, window_sec=3.0):
    """
    Compute a normalized "pulse intensity" from the last few seconds of
    green-channel values.

    The function:
      - selects recent samples within window_sec
      - normalizes them to [0, 1] based on min/max
      - returns the normalized value of the most recent sample

    Output is clamped to [0.2, 0.8] to avoid extremes. This value is used
    to scale brightness inside the ROI so the heartbeat effect is visible.
    """
    if len(buffer) < MIN_SAMPLES_FOR_PULSE:
        return 0.5  # neutral intensity before enough data is available

    now = buffer[-1][0]
    recent_vals = [v for (t, v) in buffer if now - t <= window_sec]
    if len(recent_vals) < MIN_SAMPLES_FOR_PULSE:
        return 0.5

    vals = np.array(recent_vals, dtype=np.float32)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax - vmin < 1e-3:
        # Signal is nearly flat; return neutral intensity
        return 0.5

    latest = vals[-1]
    norm = (latest - vmin) / (vmax - vmin)  # 0..1 range

    # Compress into [0.2, 0.8] to keep scaling stable
    norm = float(np.clip(0.2 + 0.6 * norm, 0.2, 0.8))
    return norm


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # HOG-based frontal face detector
    detector = dlib.get_frontal_face_detector()

    # Start webcam
    vs = VideoStream(src=CAMERA_INDEX).start()
    time.sleep(2.0)  # allow camera to warm up

    # Rolling buffer for (timestamp, green_mean)
    rppg_buffer = deque()

    try:
        while True:
            frame = vs.read()
            if frame is None:
                break

            # Convert to RGB for dlib detector
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            rects = detector(rgb, 0)
            now = time.time()

            pulse_intensity = 0.5  # default neutral value

            if len(rects) > 0:
                # Use the first detected face
                r0 = rects[0]

                # Draw face bounding box
                x1, y1, x2, y2 = r0.left(), r0.top(), r0.right(), r0.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_FACE_BOX, 2)

                # Extract forehead ROI inside the face box
                roi_box = extract_forehead_roi_from_face(frame, r0)
                if roi_box is not None:
                    x, y, w, h = roi_box
                    roi = frame[y:y + h, x:x + w]

                    # Compute mean green-channel intensity inside ROI
                    green_mean = float(np.mean(roi[:, :, 1]))

                    # Append new sample to buffer
                    rppg_buffer.append((now, green_mean))

                    # Drop samples older than the configured time window
                    while len(rppg_buffer) > 0 and (now - rppg_buffer[0][0]) > RPPG_WINDOW_SECONDS:
                        rppg_buffer.popleft()

                    # Compute normalized pulse intensity value
                    pulse_intensity = get_pulse_intensity_simple(rppg_buffer)

                    # Map pulse intensity to a brightness scale factor
                    # pulse_intensity ~0.2..0.8 --> scale ~ 0.7..1.3
                    scale = 0.7 + 1.0 * (pulse_intensity - 0.5) * 2.0

                    # Apply brightness scaling inside ROI for visualization
                    roi_float = roi.astype(np.float32)
                    roi_float *= scale
                    roi_float = np.clip(roi_float, 0, 255).astype(np.uint8)
                    frame[y:y + h, x:x + w] = roi_float

                    # Draw ROI rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_ROI_BOX, 1)

            # Text overlays
            cv2.putText(
                frame,
                f"Faces detected: {len(rects)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_TEXT,
                2,
            )

            cv2.putText(
                frame,
                f"Pulse intensity: {pulse_intensity:.2f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                2,
            )

            cv2.putText(
                frame,
                "Press 'x' to exit",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_TEXT,
                1,
            )

            cv2.imshow("Forehead rPPG Pulse Visualization", frame)

            # Exit on 'x'
            if (cv2.waitKey(1) & 0xFF) == ord("x"):
                break

    finally:
        vs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
