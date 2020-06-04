import numpy as np
import cv2
import dlib
import time
import sys

#from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QLabel, QPushButton, QProgressBar, \
    QRadioButton, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal

image_path = "faces-1.jpg"
cascade_path = "haarcascade_frontalface_default.xml"
predictor_path = "shape_predictor_68_face_landmarks.dat"

class MyWidget(QWidget):
    def __init__  (self):
        QWidget. __init__ (self)
    myclose = True

    def closeEvent(self,event):
        if self.myclose:
            print(self.myclose)
            try:
                cap.release()
                cv2.destroyAllWindows()
            except:
                print("")
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyWidget()
    w.resize(310,210)
    w.setWindowTitle('Faces')

    pbar = QProgressBar(w)
    pbar.setGeometry(10, 165, 290, 30)

    def progBarUpdate(percent):
        pbar.setValue(percent)


    def startRec():
        global cap

        starting_time = time.time()
        frame_id = 0

        faceCascade = cv2.CascadeClassifier(cascade_path)
        predictor = dlib.shape_predictor(predictor_path)

        video_file = nameEdit.text()
        try:
            cap = cv2.VideoCapture(video_file)
            nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            _, image = cap.read()
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (w, h))
            k = 0
            while True:
                _, image = cap.read()
                try:
                    h, w = image.shape[:2]
                except:
                    progBarUpdate(100)
                    break
                frame_id += 1
                progBarUpdate(100 * (k / nb_frames))
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
            #now1 = datetime.now()
            #print(now1-now)
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Файл недоступен")
            msg.setInformativeText("Проверьте корректность вводимого пути")
            msg.setWindowTitle("Ошибка чтения")
            msg.exec_()


    dirLabel = QLabel(w)
    dirLabel.setText("Расположение видеофайла:")
    dirLabel.move(10,10)
    dirLabel.show()

    nameEdit = QLineEdit(w)
    nameEdit.move(10,40)
    nameEdit.show()

    button = QPushButton(w)
    button.setText('Обработать')
    button.move(10,130)
    button.show()
    button.clicked.connect(startRec)

    w.show()
    sys.exit(app.exec_())
