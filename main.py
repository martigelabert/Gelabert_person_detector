import cv2
import numpy
import numpy as np
from pathlib import Path
#def getImages(path):


def loadImages( folder_dir : str, extension : str) -> np.ndarray:
    images = []
    _fileNames = Path(folder_dir).glob(extension)
    print(_fileNames)
    for i in _fileNames:
        img = cv2.imread(str(i), 1)
        images.append(img)
    images = np.array(images)
    return images

def averageImg(images : np.ndarray) -> np.ndarray:
    w, h, d = images[0].shape
    average  = numpy.zeros((w,h,d),numpy.float)
    for img in images:
        average = average + img / len(images)
    average = numpy.array(numpy.round(average), dtype=numpy.uint8)
    return average


def wait():
    while (True):
        # Displays the window infinitely
        key = cv2.waitKey(0)

        # Shuts down the display window and terminates
        # the Python process when a specific key is
        # pressed on the window.
        # 27 is the esc key
        # 113 is the letter 'q'
        if key == 27 or key == 113:
            break
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    empty = 'Gelabert/1660284000.jpg'
    folder_dir = 'Gelabert'
    extension = '*.jpg'
    images = loadImages(folder_dir, extension)

    img = averageImg(images)
    cv2.imshow("average",img)
    wait()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
