# Martí Gelabert Gómez

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


def resize(img):
    """Method for resizing images - DEBUG"""
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def wimgs(imgs, names, folder):
    """Method for saving a list of images given their names and folder path"""
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

    # Just the names of the files withouth the path dir attached
    names = [i.split('/')[1] for i in _fileNames]

    wimgs(images, _fileNames, 'gen/gray')

    # For the output plot
    images_color, _ = loadImages(folder_dir, extension, 1)

    # We need the iluminations of the images to be uniform
    # this way we will be able to substract the background
    # with a more consistent ilumination though the images

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    _empty = clahe.apply(cv2.imread('Gelabert/1660284000.jpg', 0))

    # Applying Adaptative Histogram Equalization
    # between the images will make the ilumination more consistent.
    images_equ = [clahe.apply(img) for img in images]
    wimgs(images_equ, _fileNames, 'gen/equ')

    # Using the empty image we obtain better results
    _empty = cv2.GaussianBlur(_empty, (15, 15), 0)

    # substraction between avg and the images with CLAHE applyed
    sub = [cv2.subtract(_empty, equ) for equ in images_equ]

    bin = [cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)[1] for s in sub]

    # Aplication of a binary mask to the already binarized images
    mask = cv2.imread('mask.png', 0) / 255.0
    mask = mask.astype(np.uint8)
    bin = [cv2.bitwise_and(b, b, mask=mask) for b in bin]

    dil = [cv2.dilate(b, np.ones((10, 10), np.uint8), iterations=1)
           for b in bin]

    # Saving images on disk
    wimgs(sub, _fileNames, 'gen/sub')
    wimgs(bin, _fileNames, 'gen/bin')
    wimgs(dil, _fileNames, 'gen/dil')

    # here we will have the contours of the multiple images we have
    contours_images = [cv2.findContours(d, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]
                       for d in dil]

    img_det = []  # images with the bboxes on top
    det = []  # the x, y, w, h for all the detections for each image

    data = dict((i, {
                                'image_name': '',
                                'rois': [],
                                'gt': []  # ground th
                    }) for i in names)

    for i in range(len(images)):
        det_frame = []
        img = images_color[i].copy()
        data[names[i]]['image_name'] = names[i]
        for c in contours_images[i]:
            x, y, w, h = cv2.boundingRect(c)
            det_frame.append([x, y, w, h])
            data[names[i]]['rois'].append((x, y, w, h))
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        det.append(det_frame)  # Set of coordenates split by frame
        img_det.append(img)  # images

    wimgs(img_det, _fileNames, 'gen/det')

    # loading the labels for the images
    df = pd.read_csv('final_labels_gelabert_person_counter.csv',
                     names=['Class', 'X', 'Y', 'filename',
                            'img_w', 'img_h'])

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
        # images_color[_fileNames.index('Gelabert/' + str(row['filename']))] = img_det[_fileNames.index('Gelabert/' + str(row['filename']))].copy()

    for index, row in df.iterrows():
        data[str(row['filename'])]['gt'].append((int(row['X']), int(row['Y'])))
    
    for i in range(len(images_color)):
        demo = images_color[i].copy()
        for gt in data[names[i]]['gt']:
            demo = cv2.circle(demo, gt, 1, (255, 0, 0), 3)
        print('real', names[i], ' -> ', len(data[names[i]]['gt']))
        plt.imshow(demo)
        plt.show()

    detection_number = np.zeros_like(contours_images)
    for i in range(len(_fileNames)):
        file = _fileNames[i].split('/')[1]
        img = images_color[i].copy()

        # Variable to count matches per image
        detections_matched = 0

        # Checking for all labels which bounding boxes contain her
        print(file)
        for coord in data[file]['gt']:
            found = False
            for x, y, w, h in det[i]:
                if w < images[0].shape[0]/3 and h < images[0].shape[1]/3:
                    if (coord[0] >= x and coord[0] <= x+w and coord[1] >= y and
                       coord[1] <= y+h):
                        # print('x = ', x, ' y = ', y, ' in ', coord)
                        detections_matched += 1
                        img = cv2.rectangle(img, (x, y), (x + w, y + h),
                                            (0, 255, 0), 2)
                        img = cv2.circle(img, coord,
                                         1, (255, 0, 0), 3)

                        # If it contains at least one label we will count it
                        # as detection we will loose all extra information
                        # but this algorithm can
                        # perform any better with the current configuration
                        found = True
                        break
                else:
                    pass

        print(detections_matched)
        detection_number[i] = detections_matched
        plt.figure(figsize=(20, 20), dpi=150)
        plt.imshow(img)
        plt.show()

    MAE = 0.0

    rows = 2
    cols = 2
    if parser.parse_args().plot:
        for i in range(len(images)):

            print("File -> ", info[i], " | ", detection_number[i], " of ",
                  info[i]['real_det'])

            MAE += abs((detection_number[i] - info[i]['real_det']))
            plt.rcParams["figure.figsize"] = (15, 15)
            fig, axs = plt.subplots(rows, cols)

            axs[0, 0].imshow(resize(images[i]), cmap='gray')
            axs[0, 0].set_title("original image")

            axs[1, 0].imshow(resize(sub[i]), cmap='gray')
            axs[1, 0].set_title("cv2.substract(_empty,original_image)")
            axs[1, 0].sharex(axs[0, 0])

            axs[0, 1].imshow(resize(cv2.cvtColor(img_det[i], 
                                    cv2.COLOR_BGR2RGB)))
            axs[0, 1].set_title("detections")

            axs[1, 1].imshow(resize(_empty), cmap='gray')
            axs[1, 1].set_title("average image")
            fig.tight_layout()
            plt.show()
    else:
        for i in range(len(images)):
            MAE += abs((detection_number[i] - info[i]['real_det']))

    print("MAE = ", MAE / len(images))
