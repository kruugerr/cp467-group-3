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


def get_pulse_intensity(buffer):
    """
    Derive a normalized "pulse" amplitude from the latest rPPG samples.
    Returns a value in roughly [0.1, 0.9], used to scale brightness.
    """
    if len(buffer) < 16:
        return 0.5  # neutral

    vals = np.array([v for (_, v) in buffer], dtype=np.float32)
    detrended = vals - np.mean(vals)
    std = np.std(detrended) + 1e-6

    latest = detrended[-1]
    z = np.clip(latest / std, -2.0, 2.0)  # -2..2
    # Map z to a 0..1-ish range
    intensity = 0.5 + 0.25 * z           # around [0,1]
    intensity = float(np.clip(intensity, 0.1, 0.9))
    return intensity


def extract_cheek_roi_from_face(frame, rect, side="left"):
    """
    Define a cheek ROI using the face bounding box rather than landmarks.
    rect: dlib rectangle
    side: "left" or "right"
    returns (x, y, w, h) or None.
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

    # Rough cheek location: mid-height, left or right third of face
    cheek_w = int(face_w * 0.22)
    cheek_h = int(face_h * 0.22)
    cheek_y = int(y1 + face_h * 0.4)

    if side == "left":
        cheek_x = int(x1 + face_w * 0.15)
    else:
        cheek_x = int(x1 + face_w * 0.63)

    # Clamp to frame
    cheek_x = max(0, min(cheek_x, w_frame - 1))
    cheek_y = max(0, min(cheek_y, h_frame - 1))
    cheek_w = min(cheek_w, w_frame - cheek_x)
    cheek_h = min(cheek_h, h_frame - cheek_y)

    if cheek_w <= 0 or cheek_h <= 0:
        return None

    return cheek_x, cheek_y, cheek_w, cheek_h


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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rects = detector(rgb, 0)
            now = time.time()

            pulse_intensity = 0.5  # default neutral

            if len(rects) > 0:
                r0 = rects[0]

                # Landmarks not strictly needed for the ROI now, but kept
                _shape = predictor(rgb, r0)
                _shape = face_utils.shape_to_np(_shape)

                roi_box = extract_cheek_roi_from_face(frame, r0, side="left")
                if roi_box is not None:
                    x, y, w, h = roi_box
                    cheek_roi = frame[y:y + h, x:x + w]

                    # Mean green channel for rPPG
                    green_mean = float(np.mean(cheek_roi[:, :, 1]))

                    # Append to buffer
                    rppg_buffer.append((now, green_mean))
                    while len(rppg_buffer) > 0 and (now - rppg_buffer[0][0]) > RPPG_WINDOW_SECONDS:
                        rppg_buffer.popleft()

                    # Get normalized pulse value for visual exaggeration
                    pulse_intensity = get_pulse_intensity(rppg_buffer)

                    # Exaggerate pulse by modulating brightness of the cheek ROI
                    # scale ~ [0.7, 1.3] depending on pulse_intensity
                    scale = 0.7 + 1.2 * (pulse_intensity - 0.5)
                    roi_float = cheek_roi.astype(np.float32)
                    roi_float *= scale
                    roi_float = np.clip(roi_float, 0, 255).astype(np.uint8)
                    frame[y:y + h, x:x + w] = roi_float

                    # draw a border around the ROI
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

                    # Analyze rPPG about once per second for HR + liveness
                    if now - last_analysis_time > 1.0:
                        last_analysis_time = now
                        hr, snr = analyze_rppg(rppg_buffer)
                        heart_rate_bpm = hr
                        rppg_live = hr is not None

                # Face box color changes with liveness
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

            # pulse_intensity itself if you want
            cv2.putText(
                frame,
                f"Pulse intensity: {pulse_intensity:.2f}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow("rPPG Cheek Pulse Demo", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("x"):
                break

    finally:
        vs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
