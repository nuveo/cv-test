"""
Automatically detect rotation and line spacing of an image of text using
Radon transform.
If image is rotated by the inverse of the output, the lines will be
horizontal (though they may be upside-down depending on the original image).
It doesn't work with black borders.
"""

import numpy
from numpy.fft import rfft
from numpy import argmax
from numpy import asarray, mean, array, blackman
from scipy import ndimage
from skimage.transform import radon
from scipy.ndimage import interpolation as inter
import warnings
import cv2


# --------------------------------------------------------
def _rms_flat(a):
    return numpy.sqrt(numpy.mean(numpy.abs(a) ** 2))

def _calculate_score(binary_image):
    hist = numpy.sum(binary_image, axis=1)
    score = numpy.sum((hist[1:] - hist[:-1]) ** 2)

    return score
# --------------------------------------------------------


def correct_rotation(I):

    # Demean: make the brightness extend above and below zero
    I = I - mean(I)  

    # Do the radon transform
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sinogram = radon(I)

    '''
    Find the RMS value of each row and find 'best' rotation,
    where the transform is lined up perfectly with the alternating dark
    text and white lines
    '''
    r = array([_rms_flat(line) for line in sinogram.transpose()])
    rotation = argmax(r)
    relative_rotation = 90 - rotation

    # Rotate image
    rotated = ndimage.rotate(I, relative_rotation, reshape=False, cval=255)
    rotated = numpy.uint16(rotated)
    rotated = cv2.bitwise_not(rotated)

    return rotated

# ----------------------------------------------------------------------

def correct_rotation_2(binary_image):
    '''
    Calculate image rotation using histogram distribution
    '''
    best_image = None
    best_value = -1
    binary_image = 255 - binary_image

    # The range is -45 to 45 to avoid finding an upside-down image
    for angle in range(-45, 45):
        rotated = inter.rotate(binary_image, angle, reshape=False, order=0)
        score = _calculate_score(rotated)

        if score > best_value:
            best_image = rotated
            best_value = score

    best_image = 255 - best_image

    return best_image