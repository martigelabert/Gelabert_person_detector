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
    print(images.shape)
    print(images[0].shape)
    average = np.zeros_like(images[0],np.float64)           
    

    for img in images:
        average = average + img 
    average = np.array(np.round(average), dtype=np.uint8)

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

def substract_all(average:np.ndarray,images:np.ndarray) -> np.ndarray:
    """Substracting the average image from all the images"""
    out = []
    for img in images:
        out.append(np.subtract(img,average))
    
    return np.array(out)    

def histogram_equalization(images:np.ndarray)-> np.ndarray:
    """Calculation of histogram equalization using openCV optimized functions"""
    output = []
    for img in images:
        output.append( cv2.equalizeHist(img))
    return np.array(output)

def check_side_by_side(img1,img2):
    res = np.hstack((img1,img2)) #stacking images side-by-side
    cv2.imshow('res.png',res)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    empty = 'Gelabert/1660284000.jpg'
    folder_dir = 'Gelabert'
    extension = '*.jpg'
    images, _fileNames = loadImages(folder_dir, extension,0)
    _empty = cv2.imread(empty,0)


    avg = averageImg(images=images)
    sub = substract_all(images=images,average=avg)
    equ = histogram_equalization(images=sub)

  
    cv2.imshow("average",avg)
    cv2.waitKey(0)

    cv2.imshow("sub",sub[0])
    cv2.waitKey(0)

    f = gabor_filter_bank(equ[0])
    f = np.array(f)

    dil = [
        cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=1)
        for img in equ
    ]

    for i in range(len(dil)):
        print("Filter n",i)
        cv2.imshow("a",dil[i])
        cv2.waitKey(0)



    for i in range(len(f)):
        print("Filter n",i)
        cv2.imshow("a",f[i])
        cv2.waitKey(0)
    
    # Generating Average Images 
    #avg = averageImg(images)
    #subs = substract_all(images=images,average=avg)



    #check_side_by_side(img1=images[0],img2=np.subtract(subs[0],_empty))    

    #cv2.imshow("empty",np.subtract(_empty,images[0]))
    #test = np.subtract(_empty,images[0]) 
    #equ = cv2.equalizeHist(test)

    #check_side_by_side(images[0],avg)


    # Now we have every frame minus the average 
    #substrations = substract_all(images=avg,average=images)

    #filtered = gabor_filter_bank(images) 
    #equ = histogram_equalization(images)

    #cv2.imshow("equ",equ[0])
    #check_side_by_side(equ[0],images[0])
    #wait()
    #a = gabor_filter_bank(images[1])[1]


        
