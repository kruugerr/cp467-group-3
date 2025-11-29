import cv2
import dlib
import time
import numpy as np
from collections import deque
from imutils.video import VideoStream
from imutils import face_utils

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# rPPG / liveness config
RPPG_WINDOW_SECONDS = 8.0      # how many seconds of data to keep
RPPG_MIN_HZ = 0.7              # ~42 bpm
RPPG_MAX_HZ = 4.0              # ~240 bpm
RPPG_MIN_DURATION = 5.0        # need at least this many seconds buffered
RPPG_MIN_SNR = 5.0             # signal-to-noise threshold
RPPG_HR_MIN_BPM = 45
RPPG_HR_MAX_BPM = 150

COLOR_NOT_LIVE = (0, 0, 255)
COLOR_LIVE = (0, 255, 0)


def analyze_rppg(buffer):
    """
    buffer: deque of (timestamp, green_mean)
    returns (heart_rate_bpm, snr) or (None, None) if not enough data / bad signal
    """
    if len(buffer) < 32:
        return None, None

    times = np.array([t for (t, v) in buffer], dtype=np.float64)
    values = np.array([v for (t, v) in buffer], dtype=np.float64)

    duration = times[-1] - times[0]
    if duration < RPPG_MIN_DURATION:
        return None, None

    # Resample to uniform time grid
    N = 256
    t0, t1 = times[0], times[-1]
    even_times = np.linspace(t0, t1, N)
    even_values = np.interp(even_times, times, values)

    # Detrend (remove mean)
    even_values = even_values - np.mean(even_values)

    # FFT
    freqs = np.fft.rfftfreq(N, d=(t1 - t0) / (N - 1))
    fft_vals = np.fft.rfft(even_values)
    psd = np.abs(fft_vals) ** 2

    # Band-pass in heart-rate range
    band_mask = (freqs >= RPPG_MIN_HZ) & (freqs <= RPPG_MAX_HZ)
    if not np.any(band_mask):
        return None, None

    freqs_band = freqs[band_mask]
    psd_band = psd[band_mask]

    peak_idx = np.argmax(psd_band)
    peak_freq = freqs_band[peak_idx]
    peak_power = psd_band[peak_idx]
    noise_power = np.mean(psd_band) + 1e-8
    snr = peak_power / noise_power

    heart_rate_bpm = peak_freq * 60.0

    if not (RPPG_HR_MIN_BPM <= heart_rate_bpm <= RPPG_HR_MAX_BPM):
        return None, None
    if snr < RPPG_MIN_SNR:
        return None, None

    return float(heart_rate_bpm), float(snr)


def extract_cheek_roi(frame, shape, side="left"):
    """
    Use a few facial landmarks to define a simple cheek bounding box.
    shape: (68, 2) landmark array
    side: "left" or "right"
    returns (x, y, w, h) or None if invalid.
    """
    if side == "left":
        # Jawline 3,4 + nose 31 + mouth corner 48
        idxs = [3, 4, 31, 48]
    else:
        # Jawline 13,12 + nose 35 + mouth corner 54
        idxs = [13, 12, 35, 54]

    pts = shape[idxs]
    x, y, w, h = cv2.boundingRect(pts)

    # Slightly shrink box to avoid edges
    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)
    x += pad_x
    y += pad_y
    w -= 2 * pad_x
    h -= 2 * pad_y

    if w <= 0 or h <= 0:
        return None

    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = min(w, w_frame - x)
    h = min(h, h_frame - y)

    if w <= 0 or h <= 0:
        return None

    return x, y, w, h


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # rPPG state
    rppg_buffer = deque()
    last_analysis_time = 0.0
    heart_rate_bpm = None
    rppg_live = False

    try:
        while True:
            frame = vs.read()
            if frame is None:
                break

            # Resize if needed for speed (optional)
            # frame = cv2.resize(frame, (640, 480))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rects = detector(rgb, 0)

            now = time.time()

            if len(rects) > 0:
                r0 = rects[0]
                shape = predictor(rgb, r0)
                shape = face_utils.shape_to_np(shape)

                # Use left cheek
                roi_box = extract_cheek_roi(frame, shape, side="left")
                if roi_box is not None:
                    x, y, w, h = roi_box
                    cheek_roi = frame[y:y + h, x:x + w]

                    # Draw ROI for visualization
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

                    # Mean green channel
                    green_mean = float(np.mean(cheek_roi[:, :, 1]))

                    # Append to buffer
                    rppg_buffer.append((now, green_mean))

                    # Drop old samples
                    while len(rppg_buffer) > 0 and (now - rppg_buffer[0][0]) > RPPG_WINDOW_SECONDS:
                        rppg_buffer.popleft()

                    # Analyze about once per second
                    if now - last_analysis_time > 1.0:
                        last_analysis_time = now
                        hr, snr = analyze_rppg(rppg_buffer)
                        heart_rate_bpm = hr
                        rppg_live = hr is not None

                # Draw face box
                x1, y1, x2, y2 = r0.left(), r0.top(), r0.right(), r0.bottom()
                box_color = COLOR_LIVE if rppg_live else COLOR_NOT_LIVE
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Overlay info
            cv2.putText(
                frame,
                f"Faces: {len(rects)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if heart_rate_bpm is not None:
                hr_text = f"HR: {heart_rate_bpm:.1f} bpm"
            else:
                hr_text = "HR: -- bpm"

            cv2.putText(
                frame,
                hr_text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            live_text = "rPPG Liveness: LIVE" if rppg_live else "rPPG Liveness: Not confirmed"
            live_color = COLOR_LIVE if rppg_live else COLOR_NOT_LIVE

            cv2.putText(
                frame,
                live_text,
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                live_color,
                2,
            )

            cv2.imshow("rPPG Liveness Demo", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("x"):
                break

    finally:
        vs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
