import cv2 as cv
import numpy as np


def _read(path):
    return cv.imread(path, cv.IMREAD_GRAYSCALE)


def _subtract(imgs, processed):
    sub = []
    for img1, img2 in zip(imgs, processed):
        subtracted = cv.subtract(img2, _read(img1))
        sub.append(cv.bitwise_not(subtracted))
    return sub


def adaptative(imgs):
    thrs = []
    for img in imgs:
        img = _read(img)
        adpt = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 20)
        thrs.append(adpt)
    return thrs


def otsu(imgs):
    ots = []
    for img in imgs:
        img = _read(img)
        t, otsu = cv.threshold(
            img, 10, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        ots.append(otsu)
    return ots


def median(imgs, k=9):
    median = []
    for img in imgs:
        img = _read(img)
        median.append(cv.medianBlur(img, k))
    return _subtract(imgs, median)


def edges(imgs, sigma=0.33):
    edg = []
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    for img_path in imgs:
        img = _read(img_path)
        # calculate median of the image
        v = np.median(img)

        # apply canny using the median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv.Canny(img, lower, upper)

        # dilate
        dilation = cv.dilate(edged, kernel, iterations=1)

        # erode
        erosion = cv.erode(dilation, kernel, iterations=1)

        inverted = cv.bitwise_not(erosion)

        edg.append(inverted)

    return edg
