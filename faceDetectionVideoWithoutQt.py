import numpy as np
import cv2
import dlib
import time
import sys

image_path = "faces-1.jpg"
cascade_path = "haarcascade_frontalface_default.xml"
predictor_path = "shape_predictor_68_face_landmarks.dat"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument('path', metavar='video_path', type=str, help='Path to source video')
    args = parser.parse_args()
    print("Source Path:", args.path)
    cap = cv2.VideoCapture(args.path)
    
    faceCascade = cv2.CascadeClassifier(cascade_path)
    predictor = dlib.shape_predictor(predictor_path)
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, image = cap.read()
    h, w = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (w, h))
    k = 0
    while True:
        _, image = cap.read()
        try:
            h, w = image.shape[:2]
        except:
            break
        k += 1

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            print
            dlib_rect
            detected_landmarks = predictor(image, dlib_rect).parts()

            landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])

                cv2.putText(image, str(idx), pos,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4,
                            color=(0, 0, 255))

                cv2.circle(image, pos, 3, color=(0, 255, 255))

        out.write(image)
        cv2.imshow("image", image)

        if ord("q") == cv2.waitKey(1):
            break

    cap.release()
    cv2.destroyAllWindows()
