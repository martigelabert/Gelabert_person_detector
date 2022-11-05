import cv2
import numpy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#def getImages(path):

def loadImages( folder_dir : str, extension : str, color = 1) -> np.ndarray:
    images = []
    _fileNames = Path(folder_dir).glob(extension)
    print(_fileNames)
    for i in _fileNames:
        img = cv2.imread(str(i), color)
        images.append(img)
    images = np.array(images)
    return images,_fileNames


def averageImg(images : np.ndarray) -> np.ndarray:
    """"Method for calculating the average between all the images avaliable"""

    average  = numpy.zeros(images[0].shape,numpy.float)
    
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

def gabor_filter_bank(img):
    """Generation of diferent Gaborn filters and apply them to a given image"""
    kernels = [
        cv2.getGaborKernel(ksize=(15, 15), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=0)
        for sigma in [3, 5, 7]
        for theta in [np.pi, np.pi / 2, 0]
        for lambd in [1.5, 2]
        for gamma in [1, 1.5]
    ]

    fig = plt.figure(figsize=(10, 12))
    
    rows = 5
    columns = len(kernels)//rows - 1
    for i in range(1, columns*rows +1):
        img = kernels[i]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

    filtered_images = [cv2.filter2D(img, cv2.CV_64F, kernel) for kernel in kernels]
    return filtered_images

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    empty = 'Gelabert/1660284000.jpg'
    folder_dir = 'Gelabert'
    extension = '*.jpg'
    images, _fileNames = loadImages(folder_dir, extension,0)

    avg = averageImg(images)
    
    
    cv2.imshow("average",images[1])
    a = gabor_filter_bank(images[1])[1]
    #np.subtract(images[1],avg)
    cv2.imshow("average",np.subtract(a,avg))
    wait()
