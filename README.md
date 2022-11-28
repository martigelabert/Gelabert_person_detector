# A simple detector for crowd counting using OpenCV and Python


## Folder structure

'match' and 'det' have the filtered rois matching the ground truth and the algorithm output respectively.

    ├── Documentation   ** Documentation with tex source
    ├── slides          ** Slides with the tex source    
    ├── Gelabert        ** images
    ├── gen
    │   ├── bin     ** binarized images
    │   ├── det     ** output 
    │   ├── dil     ** dilated images
    │   ├── equ     ** equalized images
    │   ├── gray    ** gray scale images
    │   ├── match   ** clean detections 
    │   └── sub     ** (empty - image) = sub
    ├── LAB.py      ** implementation with color background substraction
    ├── main.py     ** main program
    ├── mask.png    ** binary mask
    ├── metrics_2.csv       ** output for documentation generation
    ├── metrics.csv         ** output for documentation generation
    ├── MSE.txt             ** output for documentation generation
    ├── README.md
    └── requirements.txt        
    ...


# Execution

> Be coherent, if changed the image path you need to specify the path of the empty image

    $ python3 main.py -h
    >>>
    usage: main.py [-h] [-p | --plot | --no-plot] [-f FOLDER] [-e EXTENSION] [-em EMPTY]

    A simple program for person detectection and crowd counting in beautiful pictures!

    options:
    -h, --help            show this help message and exit
    -p, --plot, --no-plot
                            Plot the results with matplotlib at the endof the execution (default: False)
    -f FOLDER, --folder FOLDER
                            Custom path to the images
    -e EXTENSION, --extension EXTENSION
                            Custom extension to the images to load
    -em EMPTY, --empty EMPTY
                            Specify the path to the empty image

For a basic execution just :
    
    # If you want a mathplotlib window pop up with the images
    python3 main.py --plot

    # If you want just to ejecute and check directly on the gen folder
    python3 main.py

## Terminal Output

            files  precission    recall  f1 score   gt  detected  matched
    1660309200.jpg    0.336134  0.444444  0.382775   90       119       40
    1660302000.jpg    0.292308  0.368932  0.326180  103       130       38
    1660294800.jpg    0.333333  0.458333  0.385965   72        99       33
    1660320000.jpg    0.362963  0.362963  0.362963  135       135       49
    1660287600.jpg    0.200000  0.470588  0.280702   17        40        8
    1660298400.jpg    0.347368  0.311321  0.328358  106        95       33
    1660305600.jpg    0.360360  0.396040  0.377358  101       111       40
    1660316400.jpg    0.358209  0.345324  0.351648  139       134       48
    1660291200.jpg    0.352941  0.346154  0.349515   52        51       18

    MSE                      341.666667
    Macro-average precision    0.327069
    Macro-average recall       0.389344
    Macro-average F1           0.349496

# OUTPUT

Final output
![Detections](gen/det/1660291200.jpg)


Detections that match with the ground truth (NOT THE OUTPUT)
![match](gen/match/1660291200.jpg)
