import cv2
import dlib
import time
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils

# -----------------------------------------------------------------------------
# EAR-based blink / liveness configuration
# -----------------------------------------------------------------------------
# Eye Aspect Ratio (EAR) is a simple ratio between the eye's vertical opening
# and its horizontal width computed from eye landmarks. When the eye closes
# the vertical distance shrinks, so EAR drops. We use this to detect blinks
# and confirm that a real, live person is in front of the camera.
#
# New dependencies:
#   - dlib (for HOG face detector and 68-point landmark model)
#   - imutils (for VideoStream helper and landmark utilities)
#   - numpy
#
# You also need the pretrained 68-point landmark model file:
#   shape_predictor_68_face_landmarks.dat
# Download it from the official dlib model zoo and either place it next to
# this script or update PREDICTOR_PATH below to point to its location.
# -----------------------------------------------------------------------------

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# EAR below this threshold is treated as "eyes closed"
EAR_THRESHOLD = 0.21

# Number of consecutive frames with EAR below threshold required to count a blink
EAR_CONSEC_FRAMES = 3

# Number of blinks required to confirm liveness
BLINKS_REQUIRED_FOR_LIVENESS = 2

# Colors for drawing
COLOR_NOT_LIVE = (0, 0, 255)  # red
COLOR_LIVE = (0, 255, 0)      # green


def calculate_ear(eye):
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    `eye` is expected to be a 6x2 array of (x, y) coordinates corresponding
    to the 6 eye landmarks from the 68-point facial landmark model.

    EAR is a ratio of the average vertical eye opening to the horizontal
    eye width. Smaller values mean the eye is more closed.
    """
    # vertical distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])

    # Protect against division by zero
    if C == 0:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return ear


def main():
    # HOG face detector
    detector = dlib.get_frontal_face_detector()

    # 68-point facial landmark predictor (required for EAR / blink detection)
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # Pre-compute landmark index ranges for the left and right eyes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Start video camera
    vs = VideoStream(src=0).start()

    # Give web camera time to boot up before taking frame inputs
    time.sleep(2.0)

    # Blink / liveness state
    blink_count = 0
    consec_frames_closed = 0
    live_confirmed = False
    ear = 0.0

    try:
        while True:
            # Grab frame
            frame = vs.read()
            if frame is None:
                break

            # Turn frame into RGB for HOG / landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run HOG on current frame
            rects = detector(rgb, 0)

            # Reset EAR for this frame
            ear = 0.0

            # Run landmarks / EAR on the first detected face (primary subject)
            if len(rects) > 0:
                r0 = rects[0]
                shape = predictor(rgb, r0)
                shape = face_utils.shape_to_np(shape)

                # Extract eye regions and compute EAR
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = calculate_ear(leftEye)
                rightEAR = calculate_ear(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # Blink detection logic:
                # If EAR is below the threshold, increment the consecutive
                # closed-eye frame count; otherwise, if the count has been
                # high enough, register a blink.
                if ear < EAR_THRESHOLD:
                    consec_frames_closed += 1
                else:
                    if consec_frames_closed >= EAR_CONSEC_FRAMES:
                        blink_count += 1
                    consec_frames_closed = 0

                # Confirm liveness once enough blinks have been observed
                if blink_count >= BLINKS_REQUIRED_FOR_LIVENESS:
                    live_confirmed = True

            # Determine box color based on liveness
            box_color = COLOR_LIVE if live_confirmed else COLOR_NOT_LIVE

            # Draw box on faces
            for r in rects:
                # Get coords
                x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()

                # Draw colored box around faces
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Put text saying # of faces detected
            cv2.putText(
                frame,
                f"# of faces detected using HOG: {len(rects)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Overlay EAR, blink count, and liveness status
            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Blinks: {blink_count}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            liveness_text = "LIVE" if live_confirmed else "Liveness not confirmed"
            liveness_color = COLOR_LIVE if live_confirmed else COLOR_NOT_LIVE

            cv2.putText(
                frame,
                f"Liveness: {liveness_text}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                liveness_color,
                2,
            )

            cv2.imshow("HOG Face Detector", frame)

            # Refresh to get new frame every ms
            # waitKey returns ASCII value of keyboard buttons pressed
            # Interrupt condition: check for if ASCII value of x was pressed
            if (cv2.waitKey(1) & 0xFF) == ord("x"):
                break
    finally:
        vs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()