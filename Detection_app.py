import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from UI_webcam_4 import Ui_MainWindow
from models.pycuda_api import TRTEngine
from tensorrt_infer_det_without_torch import inference

class CamaraThread(QThread):
    image = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0)
        self.running = False
        if (self.capture is None) or (not self.capture.isOpened()):
            self.connect = False
        else:
            self.connect = True

    def run(self):
        while self.running and self.connect:
            ret, frame = self.capture.read()
            if ret:
                #print(1)
                self.image.emit(frame)
            else:
                print("Acquired frame fail!")
                self.connect = False
                
    def start_stop(self):
        if self.connect:
            self.running = not self.running
            if self.running:
                print("Camera is open!")
            else:
                print("Camera is close!")

    def close(self):
        if self.connect:
            self.running = False
            self.capture.release()


class ObjectDetect_MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ObjectDetect_MainWindow, self).__init__()
        self.setupUi(self)
        self.camera = CamaraThread()

        if self.camera.connect:
            self.debugBar("Camera Connection!")
        else:
            self.debugBar("Camera Disconnection!")
        self.pushButton.clicked.connect(self.ClickStartBtn)
        self.checkBox.clicked.connect(self.SetDetectMask)
        self.comboBox.currentIndexChanged.connect(self.SetViewRadio)
        self.camera.image.connect(self.UpdateImage)
        self.b_DetectMask = False
        self.b_TensorRTMode = True
        self.b_TriggerResizeView = False
        self.ViewRadio = 1.0
        self.comboBox.setCurrentIndex(2)
        self.model = YOLO("model/mask_detect/best.pt")
        results = self.model.predict(source="inference/pexels-thirdman-8482541.jpg", show=False, save=False, conf=0.5)
        self.engine = TRTEngine("model/mask_detect/best.engine")

    def ClickStartBtn(self):
        if self.camera.connect:
            icon = QIcon()
            if not self.camera.running:
                icon.addPixmap(QPixmap("icon/pause.png"), QIcon.Normal, QIcon.Off)
                self.camera.start_stop()
                self.camera.start()
            else:
                icon.addPixmap(QPixmap("icon/play.png"), QIcon.Normal, QIcon.Off)
                self.camera.start_stop()

            self.pushButton.setIcon(icon)
            self.comboBox.setEnabled(not self.camera.running)

    def SetDetectMask(self):
        if self.checkBox.isChecked():
            self.b_DetectMask = True
            #print("detect")
        else:
            self.b_DetectMask = False
            #print("not detect")

    def SetViewRadio(self):
        self.ViewRadio = float(self.comboBox.currentText())
        self.b_TriggerResizeView = True

    def UpdateImage(self, frame):
        scene = QGraphicsScene()
        frame_width = int(frame.shape[1]*self.ViewRadio)
        frame_height = int(frame.shape[0]*self.ViewRadio)
        scene.setSceneRect(0, 0, frame_width, frame_height)
        if self.b_TriggerResizeView:
            self.b_TriggerResizeView = False
            self.graphicsView.setMinimumSize(QtCore.QSize(frame_width+4, frame_height+4))

        if self.b_DetectMask:
            image = self.DetectMaks(frame)
        else:
            if self.comboBox.currentIndex() != 2:
                frame = cv2.resize(frame, (frame_width, frame_height),
                                   interpolation=cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)

        scene.addPixmap(QPixmap.fromImage(image))   
        self.graphicsView.setScene(scene)

    def DetectMaks(self, frame):
        if not self.b_TensorRTMode:
            results = self.model.predict(source=frame, show=False, save=False, conf=0.5)
            detect_image = results[0].plot()
        else:
            detect_image = inference(self.engine, frame, self.b_DetectMask)
        if self.comboBox.currentIndex() != 2:
            detect_image = cv2.resize(detect_image, (int(detect_image.shape[1]*self.ViewRadio), int(detect_image.shape[0]*self.ViewRadio)),
                                      interpolation=cv2.INTER_CUBIC)
        detect_frame = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB) # yolov8 result image is BGR(for opencv display)
        detect_frame = QImage(detect_frame, detect_image.shape[1], detect_image.shape[0], QImage.Format_RGB888)
        return detect_frame
    """
    def CloseEvent(self, event):
        if self.camera.running:
            self.camera.close()
            self.camera.terminate()
        QtWidgets.QApplication.closeAllWindows()
    """
    def debugBar(self, msg):
        self.statusbar.showMessage(str(msg), 5000)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetect_MainWindow()
    window.show()
    sys.exit(app.exec_())