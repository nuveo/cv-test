# Done in python 3.6.9
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from utils import sortPointsClockwise

def dirPath(string):

    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def filterBackground(image):

    height, width = image.shape[:2]
    #* Convert images to grayscale
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #* dilates image to filter out black text with the purpose of geting only a background image
    dilated_img = cv2.dilate(image, np.ones((5,5), np.uint8))
    #*smooths background maintaining sort of sharp background edges
    bg_img = cv2.medianBlur(dilated_img, 11)
    #*tries to remove the background from the original image and then normalize the remaining pixel values
    diff_img = 255 - cv2.absdiff(image, bg_img)
    norm_img = diff_img.copy()
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    """
    To segment the text, 2 thresholding techniques are applied: local thresholding and otsu's. The first gives a better result
    over the very dark wrinkles but maintains some of the noise of the background. The second one results in less noise but some of the dark background is merged
    into the text. Combining both segmentation yields a better result
    """
    thr_img1 = cv2.adaptiveThreshold(norm_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,10)
    _, thr_img0 = cv2.threshold(norm_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    thr_img = cv2.bitwise_or(thr_img0,thr_img1)
    
    return thr_img

def deskewImage(image):

    """
    Creates a color inverted copy of the text image to detect text angle in relation to the x-axis of the image.
    The angle detection is done trough probabilistic hough lines and they are drawn in an empty image.
    Their region is used to create a rotated bounding rectangle whose edges are used for a perspective transformation
    This acomplishes a reasonable framing of the text and ensures horizontal text alignment
    In contrast, this appoach causes a drawback related to the inclination of the letters
    """
    height, width = image.shape[:2]
    
    image_inv = image.copy()
    cv2.bitwise_not(image,image_inv)
    
    lines = cv2.HoughLinesP(image_inv, 1, np.pi/360, 100, minLineLength=width/2, maxLineGap=50)   
    line_img = np.zeros((height,width),np.uint8)
    
    for startX, startY, endX, endY in lines[:,0]:
        cv2.line(line_img,(startX,startY),(endX,endY),(255,255,255),3)
    #* Dilate lines so that the bounding rectangle doesn't crop text
    deskew_img = cv2.dilate(line_img, np.ones((25,15),np.uint8))
    
    region = cv2.findNonZero(deskew_img)
    rect = cv2.minAreaRect(region)
    points = cv2.boxPoints(rect)
    
    sorted_points = sortPointsClockwise(points)
    rect_height = np.linalg.norm([sorted_points[0], sorted_points[1]])
    rect_width = np.linalg.norm([sorted_points[1], sorted_points[2]])
    dst = np.array([[0, rect_height - 1],[0, 0],[rect_width - 1, 0],[rect_width - 1, rect_height - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(sorted_points, dst)
    
    warp = cv2.warpPerspective(image, M, (int(rect_width), int(rect_height)),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    
    return warp

def deitalicizeImage(image,scale_y = 4, min_area = 0.8, max_area = 3):

    """
    Tries to find the average angle of inclination of the letters in the text by generating contours,
    finding the angle of the rotated bounding rectangle of each contour and then shearing the image along x-axis
    to remove the inclination
    """
    
    height, width = image.shape[:2]
    mean_area = 0
    itallic_angle = 0
    counter = 0
    image_inv = image.copy()
    
    cv2.bitwise_not(image,image_inv)
    #* scales image to increase the chance that the bounding rectangle will have the same angle of the letters in the text 
    image_inv_scaled = cv2.resize(image_inv,(0,0),fx=1,fy=scale_y)

    contours, hierarchy = cv2.findContours(image_inv_scaled, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #* gets the mean area of parent contours
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if hierarchy[0,i,-1] == -1:
            mean_area += area
            counter += 1
    
    mean_area = mean_area/counter
    counter = 0
    
    #* cycles through each parent contour filters some contours around the mean area value
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > mean_area*min_area and area < mean_area*max_area and hierarchy[0,i,-1] == -1:
            rect = cv2.minAreaRect(contours[i])
            #* extracts the correct angle of each contour
            if rect[2] == 90:
                itallic_angle += rect[2]
            if rect[2] == 0:
                itallic_angle += 90
            elif rect[2] < 45 and rect[2] != 0:
                itallic_angle += 90-rect[2]
            elif rect[2] > 45 and rect[2] != 90:
                itallic_angle += 90+(90-rect[2])
            counter += 1

    #* average text angle of inclination related to the anisotropically scaled image
    itallic_angle = itallic_angle/counter*np.pi/180
    #* corrected angle for the original aspect ratio image
    real_itallic_angle = np.arctan(np.tan(itallic_angle)/scale_y)

    """
    Shear factor for correcting calculated angle
    x_new = x_old + shear_x * y_old
    y_new = y_old
    the diference between x_new and x_old is calculated with the calculated angle
    shear_x = 1/tan(real_itallic_angle)
    """
    deitalicize_shear_factor = 1/(np.tan(itallic_angle)/scale_y)
    M2 = np.float32([[1,   deitalicize_shear_factor, 0],[0, 1, 0],[0,   0, 1]])
    if deitalicize_shear_factor< 0:
        Ht = np.array([[1, 0, height*abs(deitalicize_shear_factor)], [0, 1, 0], [0, 0, 1]])
    else:
        Ht = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    #* shears and translates the image to maintain centered text
    deitalicized_image = cv2.warpPerspective(image,Ht.dot(M2), (int(width+height*abs(deitalicize_shear_factor)), height),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))

    return deitalicized_image

def main():

    parser = argparse.ArgumentParser(description='Cleans-up text images in folder.')
    parser.add_argument('--input', type=dirPath, default='noisy_data/', help='input directory')
    parser.add_argument('--save', type=bool, default=True, help='save results (True or False)')
    parser.add_argument('--output', type=dirPath, default='output/', help='output directory')
    

    args = parser.parse_args()
    for image_name in tqdm(os.listdir(args.input)):
        if image_name.endswith('.png'):
            image_path = args.input+image_name
            image = cv2.imread(image_path)
            image_filtered = filterBackground(image)
            image_aligned = deskewImage(image_filtered)
            image_deitalicized = deitalicizeImage(image_aligned)
            if args.save == False:
                cv2.imshow('filtered image', image_aligned)
                cv2.imshow('final image',image_deitalicized)
                cv2.waitKey()
            else:
                cv2.imwrite(args.output+image_name,image_deitalicized)

if __name__ == "__main__":

    main()


