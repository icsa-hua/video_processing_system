from PyQt5.QtCore import pyqtSignal, QObject
import numpy as np

class ImageUpdateSignal(QObject):
    signal = pyqtSignal(np.ndarray)