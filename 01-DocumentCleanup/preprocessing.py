import numpy as np
from PIL import Image
import os
import cv2
import imutils
import pytesseract

def save(path, img):
    cv2.imwrite(path, img)

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def median_subtract(noisy_img):
    background=cv2.medianBlur(noisy_img, 13)
    result=cv2.subtract(background, noisy_img)
    result=cv2.bitwise_not(result)
    return result

def adaptive_thresholding(img):
    img_adpt_th=cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,30)
    return img_adpt_th

def remove_background(img):
    img_from_median_filter = median_subtract(img)
    img_from_adpt_th = adaptive_thresholding(img)
    return cv2.bitwise_or(img_from_adpt_th, img_from_median_filter)

def rotate_text_with_mask(img, mask):
    # rotate text's
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(mask > 0))
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    mask_rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    mask_rotated = cv2.bitwise_not(mask_rotated)
    return rotated, mask_rotated

def remove_noisy(noisy_folder):
    for filename in os.listdir(noisy_folder):

        img = load_image(noisy_folder + '/' + filename)
        ## make a mask to rotate
        foreground = remove_background(img)
        ret2,thresh_n = cv2.threshold(foreground,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # invert color
        frame = cv2.bitwise_not(thresh_n)
        # dilate image
        kernel = np.ones((5,5),np.uint8)
        frame = cv2.dilate(frame,kernel,iterations = 1)
        frame = cv2.erode(frame,kernel,iterations = 2)
        frame = cv2.dilate(frame,kernel,iterations = 2)
        ## rotate
        rotated_text, m = rotate_text_with_mask(foreground, frame)
        kernel = np.ones((13,13),np.uint8)
        ret, rotated_text = cv2.threshold (rotated_text, 250,255, cv2.THRESH_BINARY)
        result = rotated_text

        ## test ocr
        #data = pytesseract.image_to_string(result, lang='eng', config='--psm 6')
        #print(data)
        #cv2.imshow('image', result)
        #cv2.waitKey()
        
        # italic to arial
        # shear transformations

        # show the output image
        output = "output/" + filename
        save(output, result)


if __name__ == "__main__":

    # remove noisy
    noisy_folder = "noisy_data"
    

        


    