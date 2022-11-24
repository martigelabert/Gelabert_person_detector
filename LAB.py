# Second Try
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path
import shutil
import argparse


def loadImages(folder_dir: str, extension: str, color=1) -> np.ndarray:
    """Loads the images from a selected path"""
    images = []
    _fileNames = Path(folder_dir).glob(extension)
    names = []
    for i in _fileNames:
        names.append(str(i))
        img = cv2.imread(str(i), color)
        images.append(img)
    images = np.array(images)
    # names[i] = 'Gelabert/xxxxxxx.jpg'
    return images, names


def img2lab(imgs):
    return [cv2.cvtColor(i, cv2.COLOR_BGR2LAB) for i in imgs]


def showLAB(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_LAB2RGB))
    plt.show()


def showLAB2(img1, img2):
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_LAB2RGB))
    f.add_subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_LAB2RGB))
    plt.show(block=True)


def CLAHE_overL(imagesLAB):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    output = []
    for img in imagesLAB:
        cp = img.copy()
        cp[:, :, 0] = clahe.apply(img[:, :, 0])
        output.append(cp)
    return output

# Get euclidian distance for each component
# sqrt(c1-c2 ² .....)
def substraction(images, avg):
    return [cv2.subtract(avg, img) for img in images]


def avg_imgs(images):
    return np.sum(images, 0) / len(images)


def sameImage(img1, img2):
    difference = cv2.subtract(img1, img2)
    b, g, r = cv2.split(difference)
    return cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0

def resize(img):
    """Method for resizing images - DEBUG"""
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

def normalize(data):

    info = np.finfo(data.dtype)  # Get the information of the incoming image type
    aux = data.astype(np.float64) / info.max  # normalize the data to 0 - 1
    aux = 255 * data  # Now scale by 255
    img = aux.astype(np.uint8)
    return img

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A simple program for person '
                                     + 'detectection and crowd counting in '
                                     + 'beautiful pictures!')
    parser.add_argument('-p', '--plot', type=bool, default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Plot the results with matplotlib at the end'
                              + 'of the execution')
    parser.add_argument('-f', '--folder', type=str, default='Gelabert',
                        required=False, help='Custom path to the images')
    parser.add_argument('-e', '--extension', type=str, default='*.jpg',
                        required=False,
                        help='Custom extension to the images to load')
    parser.add_argument('-em', '--empty', type=str,
                        default='Gelabert/1660284000.jpg',
                        required=False,
                        help='Specify the path to the empty image')

    folder_dir = parser.parse_args().folder
    _empty_dir = parser.parse_args().empty
    extension = parser.parse_args().extension

    images, _fileNames = loadImages(folder_dir, extension, 1)
    imagesLAB = img2lab(images)
    CLAHEimagesLAB = CLAHE_overL(imagesLAB)
    names = [i.split('/')[1] for i in _fileNames]
    # showLAB2(imagesLAB[1], CLAHEimagesLAB[1])

    # print(CLAHEimagesLAB[1][0].dtype)
    # uint8
    
    average = np.sum(CLAHEimagesLAB, 0) / len(images)
    # showLAB(average.astype(np.uint8))

    # blur_average = cv2.GaussianBlur(average, (5, 5), 0)
    blur_average = cv2.blur(average, (25, 25))
    #showLAB(blur_average.astype(np.uint8))

    CLAHEimagesLAB = [im.astype(np.float64) for im in CLAHEimagesLAB]

    foreground = substraction(CLAHEimagesLAB, blur_average)

    uintfg = [cv2.normalize(src=im, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) for im in foreground]

    gray = [cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY) for im in uintfg]

    # bin = [cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] for g in gray]
    bin = [cv2.threshold(g,200,255,cv2.THRESH_BINARY)[1] for g in gray]
    # bin = [cv2.adaptiveThreshold( g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2) for g in gray]

    #for i in range(10):
    #    plt.imshow(bin[i], cmap='gray')
    #    plt.show()    

    # Aplication of a binary mask to the already binarized images
    mask = cv2.imread('mask.png', 0) / 255.0
    mask = mask.astype(np.uint8)
    bin = [cv2.bitwise_and(b, b, mask=mask) for b in bin]

    dil = [cv2.dilate(b, np.ones((10, 10), np.uint8), iterations=1)
           for b in bin]

    # here we will have the contours of the multiple images we have
    contours_images = [cv2.findContours(d, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]
                       for d in dil]

    img_det = []  # images with the bboxes on top
    det = []  # the x, y, w, h for all the detections for each image

    data = dict((i, {
                        'image_name': '',
                        'rois': [],
                        'gt': [],  # ground th
                        'filter_rois': [],  # the rois we end up with
                        'real_det': 0,
                        'notusefull': [],
                        'n_filtered': 0,  # In case we have some bbox deleted 
                    }) for i in names)

    # Loop for fill the data dictionary with
    # the rois
    for i in range(len(images)):
        det_frame = []
        img = images[i].copy()
        data[names[i]]['image_name'] = names[i]
        for c in contours_images[i]:
            x, y, w, h = cv2.boundingRect(c)
            det_frame.append([x, y, w, h])
            data[names[i]]['rois'].append((x, y, w, h))
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        det.append(det_frame)  # Set of coordenates split by frame
        img_det.append(img)  # images

    # loading the labels for the images
    df = pd.read_csv('final_labels_gelabert_person_counter.csv',
                     names=['Class', 'X', 'Y', 'filename',
                            'img_w', 'img_h'])

    # updating the dictionary with the ground truth
    for i in range(len(images)):
        data[names[i]]['real_det'] = len(df.groupby('filename').indices[
                                            names[i]])

    for index, row in df.iterrows():
        data[str(row['filename'])]['gt'].append((int(row['X']), int(row['Y'])))

        img_det[_fileNames.index('Gelabert/'+str(row['filename']))
                ] = cv2.circle(img_det[_fileNames.index('Gelabert/'
                                                        + str(row['filename']))
                                       ],
                               (int(row['X']), int(row['Y'])),
                               1, (255, 0, 0), 3)

    filtered_imgs = []
    for i in range(len(_fileNames)):
        file = _fileNames[i].split('/')[1]
        img = images[i].copy()

        # Checking for all labels which bounding boxes contain her
        for coord in data[file]['gt']:
            for x, y, w, h in det[i]:
                # if roi not too big or really small chech it
                if ((w < images[0].shape[0]/3 and h < images[0].shape[1]/3)
                        or (w < 20 or h < 20)):
                    if (coord[0] >= x and coord[0] <= x+w and coord[1] >= y and
                       coord[1] <= y+h):
                        img = cv2.rectangle(img, (x, y), (x + w, y + h),
                                            (0, 255, 0), 2)
                        img = cv2.circle(img, coord,
                                         1, (255, 0, 0), 3)
                        data[file]['filter_rois'].append((x, y, w, h))

                        # If it contains at least one label we will count it
                        # as detection. We will loose all extra labels inside
                        # but this algorithm can't
                        # perform any better with the current configuration
                        break
                else:
                    if (x, y, w, h) not in data[file]['notusefull']:
                        data[file]['notusefull'].append((x, y, w, h))

        filtered_imgs.append(img)

    MSE = 0.0

    for i in range(len(images)):

        tp = len(data[names[i]]['filter_rois'])
        fp = len(data[names[i]]['rois']) - len(data[names[i]]['filter_rois']) - len(data[names[i]]['notusefull']) 
        precission = tp / (tp + fp)

        print("File -> ", data[names[i]]['image_name'], ' Precission = ',
                precission,
                " | ",
                (tp + fp), " of ",
                data[names[i]]['real_det'],
                ' real detections where matched', tp)

        MSE += (data[names[i]]['real_det'] - (tp + fp))**2
        if parser.parse_args().plot:

            cv2.imshow("filtered", filtered_imgs[i])
            cv2.imshow("non-filter", img_det[i])
            cv2.waitKey(0)

    MSE = MSE / (len(images))  # We are not computing the empty one
    print('MSE = ', MSE)
