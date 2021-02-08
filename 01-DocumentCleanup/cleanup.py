__author__ = 'Rafael Lopes Almeida'
__email__ = 'fael.rlopes@gmail.com'
__date__ = '07/02/2021'
'''
Remove noise from digitalized textual data.
'''

import cv2
import numpy as np
import pytesseract

from tools.utils import Utils
import tools.rotate as rotate


# ------------------------------------------------------------------------------------------
# Setting path
FOLDER_PATH = './data/input/'
OUTPUT_PATH = './output/'

files_list = Utils.file_names(FOLDER_PATH)


# ------------------------------------------------------------------------------------------
# Run
for filename in files_list:
    # ------------------------------------------------------
    # Pre-processing
    # Open image
    img = cv2.imread(FOLDER_PATH + filename)

    # Convert to gray
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold binarization
    _, img_thresh = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)

    # Auto rotate image
    img_rotated = rotate.correct_rotation(img_thresh)


    # ------------------------------------------------------
    '''
    Using pytesseract to detect words in image to  remove itallic formatting.
    Other methods should / could be implemented to avoid relying heavily 
    on pre-processing.
    '''

    # Recocnize words
    custom_config = r'--oem 3 --psm 6'  
    words = (pytesseract.image_to_string(img_rotated, config=custom_config, lang='eng')).split('\n')

    # Get text
    text = Utils.text_conversion(words)

    # Get new image
    ocr = Utils.print_text(text, fontScale=0.7, thickness=1, 
                    org_x=50, org_y=50, step=25,
                    imgx=800, imgy=500)

    # Save image to path
    cv2.imwrite(OUTPUT_PATH + filename, ocr)

    # Display image
    cv2.imshow('image', ocr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
