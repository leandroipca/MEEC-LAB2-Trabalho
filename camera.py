import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, qVersion
import cv2
import numpy as nd
#import reconhecedor_lbph


#def img2pixmap(image):
#    height, width, channel = image.shape
#    bytesPerLine = 3 * width
#    qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
#    pixmap = QPixmap.fromImage(qimage)
#    return pixmap


def grabFrame():
    if not cap.isOpened():
        cap.open(0)
        window.labelText.setText("Turning Camera ON")
    ret, image = cap.read()


    # isz=image.shape
    # ls = window.labelFrameInput.geometry()
    # window.labelFrameInput.setGeometry(10, 10, isz[1], isz[0])
    # window.labelFrameOutput.setGeometry(isz[1]+11, 10, isz[1], isz[0])

    #edges = cv2.Canny(image, 100, 200)
    #edges2 = nd.zeros(image.shape , nd.uint8)
    #edges2[:,:,0] = edges
    #edges2[:,:,1] = edges
    #edges2[:,:,2] = edges
    #window.labelFrameInput.setPixmap(img2pixmap(image)) #imagem normal
    window.labelFrameInput.setPixmap(image)  # imagem normal
    #window.labelFrameOutput.setPixmap(img2pixmap(edges2)) # imagem linhas


def on_cameraON_clicked():
    window.labelText.setText("Turning Camera ON")
    qtimerFrame.start(60)


def on_cameraOFF_clicked():
    qtimerFrame.stop()
    if cap.isOpened():
        cap.release()
    window.labelText.setText("Turning Camera OFF")


cap = cv2.VideoCapture()
app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("reconhecimento_facial.ui")
window.botaoCameraOn.clicked.connect(on_cameraON_clicked)
window.botaoCameraOff.clicked.connect(on_cameraOFF_clicked)
#window.labelFrameInput.setScaledContents(False)
window.labelFrameInput.setScaledContents(True)
window.labelFrameOutput.setScaledContents(True)

qtimerFrame = QTimer()
qtimerFrame.timeout.connect(grabFrame)

window.show()
app.exec()

"""
if __name__ == '__main__' :
    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
"""