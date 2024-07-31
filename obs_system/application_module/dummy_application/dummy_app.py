from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QThread
from PyQt5.QtGui import QPixmap, QImage
from .worker import ImageUpdateSignal
import numpy as np 


class MainWindow(QMainWindow):
    def __init__(self, object_detector):
        super().__init__()
        self.initUI()
        self.object_detector = object_detector
        self.imageUpdateSignal = ImageUpdateSignal()
        self.imageUpdateSignal.signal.connect(self.updateImageDisplay)
    
    def initUI(self):
        # Set up the main window
        self.setWindowTitle('YOLO Bounding Boxes Viewer')
        self.setGeometry(100, 100, 800, 600)
        self.label = QLabel("Image will be shown here")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        # Set up layout and widgets
        self.label = QLabel("Image will be shown here")
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def start_detection(self, patches) -> np.ndarray:
        predictions = self.object_detector.concurrent_prediction(patches)
        processed_patches = [self.object_detector.draw_boxes(patch, pred) for patch, pred in zip(patches, predictions)]
        for processed_patch in processed_patches:
            self.imageUpdateSignal.signal.emit(processed_patch)

    @pyqtSlot(np.ndarray)
    def updateImageDisplay(self, image:np.ndarray):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.label.setPixmap(pixmap)

    