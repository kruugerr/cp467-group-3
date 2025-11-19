import cv2, dlib, time
from imutils.video import VideoStream

def main():
    # HOG
    detector = dlib.get_frontal_face_detector() 

    # Start video camera
    vs = VideoStream(src=0).start()

    # Give web camera time to boot up before taking frame inputs
    time.sleep(2.0)

    try:
        while True:
            # Grab frame
            frame = vs.read()
            if frame is None: 
                break
            
            # Turn frame into RGB for HOG
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run HOG on current frame
            rects = detector(rgb, 0)

            # Draw box on faces
            for r in rects:
                # Get coords
                x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()

                # Draw green box around faces
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # Put text saying # of faces detected
            cv2.putText(frame, f"# of faces detected using HOG: {len(rects)}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            cv2.imshow("HOG Face Detector", frame)

            # Refresh to get new frame every MS 
            # waitKey returns ASCII value of keybaord buttons pressed 
            # Interrupt condition: check for if ASCII value of x was pressed 
            if (cv2.waitKey(1) & 0xFF) == ord('x'):
                break
    finally:
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()