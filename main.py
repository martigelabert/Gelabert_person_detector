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
    _empty = clahe.apply(cv2.imread(_empty_dir, 0))

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
        img = images_color[i].copy()

        # Checking for all labels which bounding boxes contain her
        for coord in data[file]['gt']:
            for x, y, w, h in det[i]:
                # if roi not too big or really small chech it
                if ((w < images[0].shape[0]/3 and h < images[0].shape[1]/3)
                        or (w < 2 or h < 2)):
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

    rows = 2
    cols = 2

    wimgs(filtered_imgs, _fileNames, 'gen/match')

    metrics = {'files': names, 'precission': [], 'recall': [], 'f1': [],
               'gt': [], 'detected': [], 'matched': []}

    for i in range(len(images)):

        if _empty_dir == 'Gelabert/'+data[names[i]]['image_name']:
            print('Not computing empty image...')
            metrics['precission'].append(0)
            metrics['recall'].append(0)
            metrics['f1'].append(0)
            metrics['gt'].append(0)
            metrics['detected'].append(0)
            metrics['matched'].append(0)
        else:
            tp = len(data[names[i]]['filter_rois'])
            fp = len(data[names[i]]['rois']) - len(data[names[i]]['filter_rois']) - len(data[names[i]]['notusefull']) 
            precission = tp / (tp + fp)

            fn = len(data[names[i]]['gt']) - len(data[names[i]]['filter_rois'])
            tn = 0
            recall = tp / (tp + fn)

            f1 = (precission*recall) / ((precission+recall)/2)

            metrics['precission'].append(precission)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['gt'].append(len(data[names[i]]['gt']))
            metrics['detected'].append(tp + fp)
            metrics['matched'].append(tp)

            MSE += (data[names[i]]['real_det'] - (tp + fp))**2
            if parser.parse_args().plot:
                plt.rcParams["figure.figsize"] = (15, 15)
                fig, axs = plt.subplots(rows, cols, dpi=150)

                axs[0, 0].imshow(resize(images[i]), cmap='gray')
                axs[0, 0].set_title("original image")

                axs[1, 0].imshow(resize(sub[i]), cmap='gray')
                axs[1, 0].set_title("cv2.substract(_empty,original_image)")
                axs[1, 0].sharex(axs[0, 0])

                axs[0, 1].imshow(resize(cv2.cvtColor(filtered_imgs[i],
                                        cv2.COLOR_BGR2RGB)))
                axs[0, 1].set_title("Clean detections")

                axs[1, 1].imshow(resize(cv2.cvtColor(img_det[i],
                                        cv2.COLOR_BGR2RGB)))
                axs[1, 1].set_title("Non-filtered detections")
                fig.tight_layout()
                plt.show()

    print(pd.DataFrame.from_dict(metrics))
    MSE = MSE / (len(images)-1)  # We are not computing the empty one
    print('MSE = ', MSE)
