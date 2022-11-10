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
    return images, _fileNames


def averageImg(images : np.ndarray) -> np.ndarray : 
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
        cv2.getGaborKernel(ksize=(15, 15), sigma=sigma, theta=theta, 
            lambd=lambd, gamma=gamma, psi=0)

        for sigma in [3, 5, 7]
        for theta in [np.pi, np.pi / 2, 0]
        for lambd in [1.5, 2]
        for gamma in [1, 1.5]
    ]

    if show:
        fig = plt.figure(figsize=(10, 12))
        
        rows = 5
        columns = len(kernels)//rows - 1
        for i in range(1, columns*rows + 1):
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

    # This way the persons are more whity
    subs = [cv2.subtract(avg_lab, lab) for lab in labs]

    # It is normal to get this blueish color ?
    return [cv2.cvtColor(img, cv2.COLOR_Lab2BGR) for img in subs]


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


def Method01(folder_dir, extension):
    """Method01 where the image processing is done in black and white"""
    images, _fileNames = loadImages(folder_dir, extension, 0)

    # Applying Histogram Equalization, this way the iluminations
    # between the images will be more consistent.
    images_equ = [cv2.equalizeHist(img) for img in images] 

    # Image Averaging
    avg = images_equ[0]
    for i in range(len(images_equ)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg = cv2.addWeighted(images_equ[i], alpha, avg, beta, 0.0)
    
    # substraction
    sub = [cv2.subtract(avg, equ) for equ in images_equ]
    
    cv2.imshow("Method 1, sub", sub[0])
    bin = [cv2.threshold(s, 160, 255, cv2.THRESH_BINARY)[1]
        for s in sub]
    
    dil = [cv2.dilate(b, np.ones((5, 5), np.uint8), iterations=1)for b in bin]
     
    # here we have the contours of multiple images
    contours_images = [cv2.findContours(d, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] for d in dil]
    cv2.imshow("Method 1, dil", dil[0])
    
    img_det = []
    for i in range(len(contours_images)):
        img = images[i]
        for c in contours_images[i]:
            x,y,w,h = cv2.boundingRect(c)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        img_det.append(img)
   
    for i in range(len(img_det)):
        scale_percent = 60 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
    
        # resize image
        resized = cv2.resize(img_det[i], dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("out",resized)
        cv2.waitKey(0)

def scale(img):
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 


def Method02(folder_dir, extension):
    """Method02 where the image processing is done in color"""
    # the alpha channel is dropped
    images, _fileNames = loadImages(folder_dir, extension, 1)

    # Conversion to LAB
    images_lab = [cv2.cvtColor(im, cv2.COLOR_BGR2LAB) for im in images]
    
    im_equ = [] 
    # Aplication of histogram equalization on L
    for im in images_lab:
        L, A, B = cv2.split(im)
        L_equ = cv2.equalizeHist(L)
        _im = cv2.merge((L_equ, A, B))
        im_equ.append(_im)
    
    # Conversion to avoid overflow
    im_equ = np.float64(im_equ)
     
    avg = im_equ[0]
    for i in range(len(im_equ)):
        if i == 0:
            pass
        else:
            avg += im_equ[i]
    
    avg = avg/len(im_equ)
 
    # https://stackoverflow.com/questions/35668074/how-i-can-take-the-average-of-100-image-using-opencv
    # cv2.imshow("method2 avg", cv2.cvtColor(np.uint8(avg),cv2.COLOR_LAB2BGR))
    
    # blur with gaussian kernels, need odd ksize
    avg =  cv2.GaussianBlur(avg, (17, 17), 0)
    avg_gray = cv2.cvtColor(cv2.cvtColor(np.uint8(avg), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow("avg gaussian applyed method 2", scale(avg_gray))
    cv2.imshow("equalized img", cv2.cvtColor(np.uint8(im_equ[0]), cv2.COLOR_LAB2BGR))

    # This is now on uint8
    # cv2.imshow("LAB avg_gray", avg_gray) 
    #print(avg_gray.dtype)  
    
 #   print(np.uint8(im_equ[0].shape))
 #   a = cv2.cvtColor(np.int(im_equ[0]), cv2.COLOR_LAB2BGR)
 #   cv2.imshow("aaa",np.uint8(a))

    #general  = [ cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY) 
    #        for img in im_equ]
     
    #cv2.imshow("LAB general", general[0])
    
    # cv2.imshow("method2 avg", cv2.cvtColor(np.uint8(avg),cv2.COLOR_LAB2BGR))
   
    # Conversion to gray
    # avg_gray = cv2.cvtColor(cv2.cvtColor(np.uint8(avg), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)
    # im_equ_gray  = [ cv2.cvtColor(cv2.cvtColor(np.uint8(im), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY) for im in im_equ]
   
    
    # substraction
    # sub = [cv2.subtract( im, avg_gray) for im in im_equ_gray]
    
    # cv2.imshow("method2 sub", im_equ[0])
    
# TODO : Check this web https://answers.opencv.org/question/230058/opencv-background-subtraction-get-color-objects-python/


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    empty = 'Gelabert/1660284000.jpg'
    folder_dir = 'Gelabert'
    extension = '*.jpg'
    # Load the images with color or not.

    images, _fileNames = loadImages(folder_dir, extension, cv2.IMREAD_COLOR)
    _empty = cv2.imread(empty, cv2.IMREAD_COLOR)

    Method02(folder_dir, extension)
    
    # TODO: How I do an histogram equalization if I need to work
    # with the color images to not loose a lot of information?

    # We need the iluminations of the images to be uniform
    # this way we will be able to substract the background
    # Therefore, we need to execute equalizeHist to make
    # everything more uniform
    avg = averageImg(images=images)
    # desmarcar
    #cv2.imshow("average", avg)
    
    sub = substract_all(images=images, average=avg)

    

    sub_gray = [cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) for s in sub]    
  
    # TODO : Improve the binarization and check the normalization
    # what I should do?

    # Maybe Check the two
    #

    bin = [cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
        for gray in sub_gray]
    
    # desmarcar
    #cv2.imshow("binarized image", bin[0])

    # Remeberb this is with dilate applyed...
    # cv2.imshow("bin with dilation applyed",  cv2.dilate(bin[0], np.ones((5, 5), np.uint8), iterations=1))
    
    cv2.imwrite("test/1imagen_original.png", images[0])
    cv2.imwrite("test/2imagen_average.png", avg)
    cv2.imwrite("test/3imagen_sub.png", sub[0])
    cv2.imwrite("test/4imagen_bin.png", bin[0])
    
    
    # equ = histogram_equalization(images=images)
    cv2.waitKey(0)
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

        
