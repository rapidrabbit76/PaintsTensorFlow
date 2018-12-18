import cv2, numpy as np


def convertQImageToMat(incomingImage):
    '''  Converts a QImage into an opencv MAT format  '''
    try:

        incomingImage = incomingImage.convertToFormat(4)
        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
        img = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        return img

    except AttributeError:
        pass
