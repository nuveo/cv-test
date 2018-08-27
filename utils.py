import cv2
import numpy as np
from scipy.signal import convolve2d
from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


def Kuwahara(original, winsize):
    """
    Kuwahara filters an image using the Kuwahara filter

    Inputs:
    original      -->    image to be filtered
    windowSize    -->    size of the filter window: legal values are
                                                    5, 9, 13, ... = (4*k+1)
    This function is optimised using vectorialisation, convolution and the fact
    that, for every subregion
        variance = (mean of squares) - (square of mean).
    A nested-for loop approach is still used in the final part as it is more
    readable, a commented-out, fully vectorialised version is provided as well.

    Example
    filtered = Kuwahara(original,5);
    Filter description:
    The Kuwahara filter works on a window divided into 4 overlapping
    subwindows (for a 5x5 pixels example, see below). In each subwindow, the mean and
    variance are computed. The output value (located at the center of the
    window) is set to the mean of the subwindow with the smallest variance.

        ( a  a  ab   b  b)
        ( a  a  ab   b  b)
        (ac ac abcd bd bd)
        ( c  c  cd   d  d)
        ( c  c  cd   d  d)

    References:
    http://www.ph.tn.tudelft.nl/DIPlib/docs/FIP.pdf
    http://www.incx.nec.co.jp/imap-vision/library/wouter/kuwahara.html


    Copyright Luca Balbi, 2007
    Original license is contained in a block comment at the bottom of this file.

    Translated from Matlab into Python by Andrew Dussault, 2015
    """

    # Check the time:
    # t1=time.time()

    # image = original.copy()
    # make sure original is a numpy array
    image = original.astype(np.float64)
    # make sure window size is correct

    if winsize % 4 != 1:
        raise Exception("Invalid winsize %s: winsize must follow formula: w = 4*n+1." % winsize)

    # Build subwindows
    tmpAvgKerRow = np.hstack((np.ones((1, int((winsize - 1) / 2 + 1))), np.zeros((1, int((winsize - 1) / 2)))))
    tmpPadder = np.zeros((1, winsize))
    tmpavgker = np.tile(tmpAvgKerRow, (int((winsize - 1) / 2 + 1), 1))
    tmpavgker = np.vstack((tmpavgker, np.tile(tmpPadder, (int((winsize - 1) / 2), 1))))
    tmpavgker = tmpavgker / np.sum(tmpavgker)

    # tmpavgker is a 'north-west' subwindow (marked as 'a' above)
    # we build a vector of convolution kernels for computing average and
    # variance
    avgker = np.empty((4, winsize, winsize))  # make an empty vector of arrays
    avgker[0] = tmpavgker  # North-west (a)
    avgker[1] = np.fliplr(tmpavgker)  # North-east (b)
    avgker[2] = np.flipud(tmpavgker)  # South-west (c)
    avgker[3] = np.fliplr(avgker[2])  # South-east (d)

    # Create a pixel-by-pixel square of the image
    squaredImg = image ** 2

    # preallocate these arrays to make it apparently %15 faster
    avgs = np.zeros([4, image.shape[0], image.shape[1]])
    stddevs = avgs.copy()

    # Calculation of averages and variances on subwindows
    for k in range(4):
        avgs[k] = convolve2d(image, avgker[k], mode='same')  # mean on subwindow
        stddevs[k] = convolve2d(squaredImg, avgker[k], mode='same')  # mean of squares on subwindow
        stddevs[k] = stddevs[k] - avgs[k] ** 2  # variance on subwindow

    # Choice of index with minimum variance
    indices = np.argmin(stddevs, 0)  # returns index of subwindow with smallest variance

    # Building the filtered image (with nested for loops)
    filtered = np.zeros(original.shape)
    for row in range(original.shape[0]):
        for col in range(original.shape[1]):
            filtered[row, col] = avgs[indices[row, col], row, col]

    # filtered=filtered.astype(np.uint8)
    return filtered.astype(np.uint8)

def checkIfPointInsideArea(mask, bbox):
    """"" 
    Check how many corners of a bouding box is inside of a polygon. 
    :param polygon: A list with points that generates a polygon. [(x,y),(x2,y2),(x3,y3)...]
    :param bbox: A bounding box. [x1,y1,x2,y2]
    :return list_bool: A list with bool variables telling which points from bbox are inside the polygon

    """""
    points = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
    list_bool = []
    for point in points:
        dis = cv2.pointPolygonTest(np.array(mask, np.int32), tuple(point), True)
        # print(dis)
        isInside = dis > 0
        list_bool.append(isInside)
    return list_bool

def text_dilate(img):
    """"" 
    Apply morph process
    :param img: Numpy array image.
    :return img2: Dilated image.

    """""
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode( img, kernel, iterations=1)
    kernel = np.ones((3, 4), np.uint8)
    img = cv2.dilate( img, kernel, iterations=10)
    return img

def area(a, b):
    """"" 
    Calculate the intersection area from two rectagles.
    :param a: A object rectagle.[xmin,ymin,xmax,ymax]
    :param b: A object rectagle.[xmin,ymin,xmax,ymax]
    :return dx*dy: Intersection area

    """""
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy

def deskew(img):
    """"" 
    Verify the angle of the objects in the image and deskew .
    :param img: Numpy array image.
    :return rotated: Fixed image 

    """""
    bit = cv2.bitwise_not(img.copy())
    thresh = cv2.threshold(bit, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def denoising(img,min_size):
    """"" 
    Remove noise from a image taking in account the size of the 
    contour found.
    :param img: Numpy array image.
    :return img2: Image with noise removed. 

    """""
    connectivity = 4
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    img2 = np.zeros((output.shape), np.uint8)

    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2

