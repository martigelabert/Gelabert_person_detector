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


def resize(img):
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def wimgs(imgs, names, folder):
    """Method for printing dat"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)

    for i in range(len(imgs)):
        cv2.imwrite(os.path.join(folder, names[i].split('/')[1]), imgs[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple program for person '
                                     + 'detectection and crowd counting in '
                                     + 'beautiful pictures!')
    parser.add_argument('-p', '--plot', type=bool, default=False,
                        help='Plot the results with matplotlib at the end'
                              + 'of the execution')
    parser.add_argument('-f', '--folder', type=str, default='Gelabert',
                        required=False, help='Custom path to the images')
    parser.add_argument('-e', '--extension', type=str, default='*.jpg',
                        required=False,
                        help='Custom extension to the images to load')

    folder_dir = parser.parse_args().folder
    extension = parser.parse_args().extension

    # Load the images in gray scale.
    images, _fileNames = loadImages(folder_dir, extension, 0)
    # For the output plot
    images_color, _ = loadImages(folder_dir, extension, 1)
    _empty = cv2.imread('Gelabert/1660284000.jpg', 0)

    # We need the iluminations of the images to be uniform
    # this way we will be able to substract the background
    # with a more consistent ilumination though the images

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Applying Adaptative Histogram Equalization
    # between the images will make the ilumination more consistent.
    images_equ = [clahe.apply(img) for img in images] 

    # Image Averaging
    avg = images_equ[0].astype(np.float64)  # We use float64 to avoid overflow
    for i in range(len(images_equ)):
        if i == 0:
            pass
        else:
            avg += images_equ[i].astype(np.float64)

    avg = avg/len(images_equ)
    avg = avg.astype(np.uint8)

    # blur with gaussian kernels, need odd ksize
    # avg = cv2.GaussianBlur(avg, (17, 17), 0)
    # Using the empty image we obtain better results
    avg = cv2.GaussianBlur(_empty, (15, 15), 0)

    # substraction between avg and the images with CLAHE applyed
    sub = [cv2.subtract(avg, equ) for equ in images_equ]
    sub = [cv2.erode(b, np.ones((1, 1), np.uint8), iterations=1)
           for b in sub]

    # cv2.imshow("Method 1, sub", sub[0])
    # bin = [cv2.threshold(s, 127, 255, cv2.THRESH_BINARY)[1] for s in sub]
    bin = [cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)[1] for s in sub]
    # bin = [cv2.threshold(s, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #       for s in sub]

    # Aplication of a binary mask to the already binarized images
    mask = cv2.imread('mask.png', 0) / 255.0
    mask = mask.astype(np.uint8)
    bin = [cv2.bitwise_and(b, b, mask=mask) for b in bin]

    dil = [cv2.dilate(b, np.ones((10, 10), np.uint8), iterations=1)
           for b in bin]

    wimgs(sub, _fileNames, 'gen/sub')
    wimgs(bin, _fileNames, 'gen/bin')
    wimgs(dil, _fileNames, 'gen/dil')

    # here we will have the contours of the multiple images we have
    contours_images = [cv2.findContours(d, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]
                       for d in dil]

    # cv2.imshow("Method 1, dil", dil[0])

    img_det = []  # images with the bboxes on top
    det = []  # the x, y, w, h for all the detections for each image
    for i in range(len(contours_images)):
        det_frame = []
        img = images_color[i].copy()
        for c in contours_images[i]:
            x, y, w, h = cv2.boundingRect(c)
            det_frame.append([x, y, w, h])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        det.append(det_frame)  # Set of coordenates split by frame
        img_det.append(img)  # images

    # loading the labels for the images
    df = pd.read_csv('final_labels_gelabert_person_counter.csv',
                     names=['Class', 'X', 'Y', 'filename',
                            'img_w', 'img_h'])

    # for i in df.groupby('filename').indices:
    #    print(df.groupby('filename').indices[i])

    rows = 2
    cols = 2

    info = []
    for i in range(len(images)):
        x = {
             'name': _fileNames[i],
             #   'original': images[i],
             #   'substracted': sub[i],
             #   'detection_img': img_det[i],
             # 'det_bbox': det[i],
             'num_det': len(det[i]),
             'real_det': len(df.groupby('filename').indices[_fileNames[i]
                             .split('/')[1]])
            }
        info.append(x)

    for index, row in df.iterrows():
        img_det[_fileNames.index('Gelabert/'+str(row['filename']))
                ] = cv2.circle(img_det[_fileNames.index('Gelabert/'
                                                        + str(row['filename']))
                                       ], (int(row['X']), int(row['Y'])),
                               1, (255, 0, 0), 3)

    MAE = 0.0

    if parser.parse_args().plot:
        for i in range(len(images)):
            print("File -> ", info[i], " | ", info[i]['num_det'], " of ",
                  info[i]['real_det'])
            MAE += abs((info[i]['num_det'] - info[i]['real_det']))
            plt.rcParams["figure.figsize"] = (15, 15)
            fig, axs = plt.subplots(rows, cols)

            axs[0, 0].imshow(resize(images[i]), cmap='gray')
            axs[0, 0].set_title("original image")

            axs[1, 0].imshow(resize(sub[i]), cmap='gray')
            axs[1, 0].set_title("cv2.substract(avg,original_image)")
            axs[1, 0].sharex(axs[0, 0])

            axs[0, 1].imshow(resize(cv2.cvtColor(img_det[i], 
                                    cv2.COLOR_BGR2RGB)))
            axs[0, 1].set_title("detections")

            axs[1, 1].imshow(resize(avg), cmap='gray')
            axs[1, 1].set_title("average image")
            fig.tight_layout()
            plt.show()
    else:
        for i in range(len(images)):
            MAE += abs((info[i]['num_det'] - info[i]['real_det']))

    cv2.imwrite("gen/avg_blur.png", avg)
    print("MAE = ", MAE / len(images))
