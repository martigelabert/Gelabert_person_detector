import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


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
   
    avg_image = images[0]
    for i in range(len(images)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(images[i], alpha, avg_image, beta, 0.0)

    return avg_image


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


def gabor_filter_bank(image, show=False):
    """Generation of diferent Gaborn filters and apply them to a given image"""
    kernels = [
        cv2.getGaborKernel(ksize=(15, 15), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=0)
        for sigma in [3, 5, 7]
        for theta in [np.pi, np.pi / 2, 0]
        for lambd in [1.5, 2]
        for gamma in [1, 1.5]
    ]

    if show:
        fig = plt.figure(figsize=(10, 12))
        
        rows = 5
        columns = len(kernels)//rows - 1
        for i in range(1, columns*rows +1):
            img = kernels[i]
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()
    
    # Extraction of features
    filtered_images = [cv2.filter2D(image, cv2.CV_64F, kernel) for kernel in kernels]
    return filtered_images


def substract_all(average: np.ndarray, images: np.ndarray) -> np.ndarray:
    """Substracting the average image from all the images"""
        
    # We will need to transform to LAB, this way we will have
    # a better representation of the substraction than
    # the RGB simple one (loose of information)
    
    labs = [cv2.cvtColor(img, cv2.COLOR_BGR2Lab) for img in images]
    avg_lab = cv2.cvtColor(average, cv2.COLOR_BGR2Lab)
    
    print('Conversion OK')

    subs = [np.subtract(lab, avg_lab) for lab in labs]

    return  [cv2.cvtColor(img, cv2.COLOR_Lab2BGR) for img in subs]


def histogram_equalization(images: np.ndarray) -> np.ndarray:
    """Calculation of histogram equalization using openCV optimized functions"""
    output = []
    for img in images:
        output.append( cv2.equalizeHist(img))
    return np.array(output)


def check_side_by_side(img1, img2):
    res = np.hstack((img1, img2))
    # stacking images side-by-side
    cv2.imshow('res.png', res)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    empty = 'Gelabert/1660284000.jpg'
    folder_dir = 'Gelabert'
    extension = '*.jpg'
    # Load the images with color or not.

    images, _fileNames = loadImages(folder_dir, extension, 1)
    _empty = cv2.imread(empty, 1)
 
    # TODO: How I do an histogram equalization if I need to work
    # with the color images to not loose a lot of information?

    # We need the iluminations of the images to be uniform
    # this way we will be able to substract the background
    # Therefore, we need to execute equalizeHist to make
    # everything more uniform
    avg = averageImg(images=images)
    cv2.imshow("average", avg)
    
    sub = substract_all(images=images, average=avg)
    cv2.imshow("substraction", sub[0])
    cv2.waitKey(0)
    # equ = histogram_equalization(images=images)

    ############################################

    # cv2.imshow("average",avg)
    # cv2.waitKey(0)

    # cv2.imshow("sub",sub[0])
    # cv2.waitKey(0)

    # Expand the whites
    # dil = [
    #    cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
        # cv2.erode(img, np.ones((5, 5), np.uint8), iterations=1)
    #    for img in equ
    # ]

    # 32 is clean
    # f = gabor_filter_bank(dil[0])
    # f = np.array(f)

    # for i in range(len(f)):
    #    print("Filter n", i)
    #    cv2.imshow("a", f[i])
    #    cv2.waitKey(0)

        
