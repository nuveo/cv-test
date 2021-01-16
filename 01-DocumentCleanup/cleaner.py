import cv2
import matplotlib.pyplot as plt
import numpy as np


def clean_document(document):
    """
    Removes the noisy background from documents and returns only the relevant part (text).
    :param document: BGR image with the document
    :return: binary image with background in 255 and text in 0
    """
    new_image = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
    new_image = cv2.adaptiveThreshold(new_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 30)

    return new_image