import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import glob


def loadImages(folder_dir: str, extension: str, color=1) -> np.ndarray:
    """Allows loading the images from a selected path"""
    images = []
    _fileNames = Path(folder_dir).glob(extension)
    names = []
    for i in _fileNames:
        names.append(str(i))
        img = cv2.imread(str(i), color)
        images.append(img)
    images = np.array(images)

    return images, names


def check_side_by_side(img1, img2):
    res = np.hstack((img1, img2))
    # stacking images side-by-side
    cv2.imshow('res.png', res)


def resize(img):
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    empty = 'Gelabert/1660284000.jpg'
    folder_dir = 'Gelabert'
    extension = '*.jpg'
    
    # Load the images in gray scale.
    images, _fileNames = loadImages(folder_dir, extension, 0)
    _empty = cv2.imread(empty, cv2.IMREAD_COLOR)
 
    # We need the iluminations of the images to be uniform
    # this way we will be able to substract the background
    # with a more consistent ilumination though the images

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Applying Adaptative Histogram Equalization
    # between the images will make the ilumination more consistent.
    images_equ = [clahe.apply(img) for img in images] 

    # Image Averaging
    avg = images_equ[0].astype(np.float64) 
    for i in range(len(images_equ)):
        if i == 0:
            pass
        else:
            avg += images_equ[i].astype(np.float64) 
    
    avg = avg/len(images_equ)
    avg = avg.astype(np.uint8)
 
    # blur with gaussian kernels, need odd ksize
    avg = cv2.GaussianBlur(avg, (17, 17), 0)
 
    # substraction
    sub = [cv2.subtract(avg, equ) for equ in images_equ]
     
    # cv2.imshow("Method 1, sub", sub[0])
    bin = [cv2.threshold(s, 127, 255, cv2.THRESH_BINARY)[1] for s in sub]
    
    dil = [cv2.dilate(b, np.ones((10, 10), np.uint8), iterations=1)
           for b in bin]
     
    # here we will have the contours of the multiple images we have
    contours_images = [cv2.findContours(d, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)[0] 
                       for d in dil]
    
    # cv2.imshow("Method 1, dil", dil[0])
    
    img_det = []  # images with the bboxes on top
    det = []  # the x, y, w, h for all the detections for each image
    for i in range(len(contours_images)):
        det_frame = []
        img = images[i].copy()
        for c in contours_images[i]:
            x, y, w, h = cv2.boundingRect(c)
            det_frame.append([x, y, w, h])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        det.append(det_frame)
        img_det.append(img)

    # loading the labels for the images
    df = pd.read_csv('labels_12112022-labels_2022-11-12-01-50-32.csv') 
    print(df)

    rows = 2
    cols = 2
        
    dictionary = {}
    for i in range(len(images)):
       
        fig, axs = plt.subplots(rows, cols)
        
        axs[0, 0].imshow(resize(images[i]), cmap='gray')
        axs[0, 0].set_title("original image")
        
        axs[1, 0].imshow(resize(sub[i]), cmap='gray')
        axs[1, 0].set_title("cv2.substract(avg,original_image)")
        axs[1, 0].sharex(axs[0, 0])
        
        axs[0, 1].imshow(resize(img_det[i]), cmap='gray')
        axs[0, 1].set_title("detections")
        
        axs[1, 1].imshow(resize(avg), cmap='gray')
        axs[1, 1].set_title("average image")
        fig.tight_layout()
        plt.show()

        x = {
             'name': _fileNames[i],
             #   'original': images[i],
             #   'substracted': sub[i],
             #   'detection_img': img_det[i],
             'det_bbox': det[i]
            }
        dictionary.update(x)



