import cv2
import numpy as np
from scipy.ndimage import interpolation as inter


def clean_document(document):
    """
    Removes the noisy background from documents and returns only the relevant part (text).
    :param document: BGR image with the document
    :return: binary image with background in 255 and text in 0
    """
    new_image = binarize_image(document)
    new_image = correct_skew(new_image)
    new_image = remove_noise_based_on_histogram(new_image)
    return new_image


def binarize_image(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_image = cv2.adaptiveThreshold(new_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 30)
    return new_image


def correct_skew(binary_image):
    """
    Rotates the image in every angle between [-45, 45] (with step 1)
    and returns the version with best skew score.
    Further reading: https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7
    :param binary_image: Image with bg in 255 and text in 0
    :return: An deskewed image with bg in 255 and text in 0
    """
    best_image = None
    best_value = -1
    binary_image = invert_binary_image(binary_image)
    for angle in range(-45, 45):  # The range is -45 to 45 to avoid finding an upside-down image
        rotated = inter.rotate(binary_image, angle, reshape=False, order=0)
        score = calculate_skew_score(rotated)
        if score > best_value:
            best_image = rotated
            best_value = score

    best_image = invert_binary_image(best_image)

    return best_image


def calculate_skew_score(binary_image):
    """
    Calculates the skew score to a certain image. The higher the better.
    Further reading: https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7
    :param binary_image: Image with bg in 255 and text in 0
    :return: An int with the score saying how aligned is binary_image
    """
    hist = np.sum(binary_image, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return score


def remove_noise_based_on_histogram(binary_image):
    binary_image = invert_binary_image(binary_image)
    hist = np.sum(binary_image, axis=1)
    for line, h in enumerate(hist):
        if (h / 255) < (binary_image.shape[0]*.05):
            binary_image[line, :] = 0
    binary_image = invert_binary_image(binary_image)
    return binary_image


def invert_binary_image(binary_image):
    binary_image = 255 - binary_image
    return binary_image