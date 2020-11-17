import cv2
import numpy as np
import pywt as pywt
import sys

def show(image):
    """
    Just a function to visualize the image

    :param image:
    """
    cv2.imshow('image', image)
    while True: 
        a = cv2.waitKey(0)
        if a==27: #press esc to exit
            break
        elif a==113: # press q to kill the process
            exit(0)


def w2d(img, mode='haar', level=20):
    """
    Discrete Wavelet transform. This function analyse the image in the frequence domain to separate the componets better
    
    :param img: Image to be transformed
    :param mode: Wavelet chosen. Default=haar, this is a square-shaped wavelet
    :param level: An integer which says the level of decomposition in the image
    """

    img =  np.float32(img)   
    img = img/255.0

    #wavelet coefficients
    coeffs=pywt.wavedec2(img, mode, level=level)

    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # Inverse Discrete Wavelet Transform
    img_H=pywt.waverec2(coeffs_H, mode);
    img_H *= 255;
    img_H =  np.uint8(img_H)

    return(img_H)

def color_quantization(img, K=4):
    """
    reduce the number of collor in the image using k-means algorithm.
    
    :param K: number of clusters (default =4, the better I found)
    """
    
    # convert to np.float32
    pixel_values = img.reshape((-1, 2))

    Z = np.float32(pixel_values)

    stoping_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1)

    _,label,centers=cv2.kmeans(Z,K,None,stoping_criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Go back to the original image
    centers = np.uint8(centers)

    reshape = centers[label.flatten()]
    reshaped = reshape.reshape((img.shape))

    # do some initial filters
    unique, counts = np.unique(reshaped, return_counts=True)
    unique.sort()

    reshaped[reshaped<=unique[0]] = 255
    reshaped[reshaped>=unique[len(unique)-1]] = 255

    return reshaped

def correct_rotation(image):
    """
    This is a simple function to get the rect where the text are and detect the angle of this rect to rotate the image.
    """

    rect = np.column_stack(np.where(image < 255))
    angle = cv2.minAreaRect(rect)[-1] #return angles between [-90,0)

    #The angles are inversed because they default rotation direction are in clockwise.
    if angle < -45: # a correction when the angle is <-45
        angle = -(90 + angle)

    else:
        angle = -angle

    h, w = image.shape[:2]
    c_w = w // 2
    c_h = h // 2
    M = cv2.getRotationMatrix2D((c_w, c_h), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def main(image_path):
         
    img = cv2.imread(image_path, 0)
    if img is None:
        return None
    
    img_copy = img.copy()

    # wavelet transform and color quantization to get the initial image to segmentate
    img = w2d(img)
    img = color_quantization(img)

    # Some morphological transformations to clean the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    
    img = cv2.GaussianBlur(img,(3,3),0)

    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    _, img = cv2.threshold(img, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # The lines of the image are initialy extracted
    new_img = np.zeros(img.shape, dtype=int)
    new_img+=255
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]==0:
                new_img[i][j] = img_copy[i][j]

    new_img = cv2.convertScaleAbs(new_img)
    
    # by now some refinements in the result

    #this kernel will highlight mainly the horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    img = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel, iterations=2)
    _, img = cv2.threshold(img, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    

    new_img = np.zeros(img.shape, dtype=int)
    new_img+=255
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]==0:
                new_img[i][j] = img_copy[i][j]

    new_img = cv2.convertScaleAbs(new_img)
    img_result = cv2.adaptiveThreshold(new_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,1)

    img_final = correct_rotation(img_result)

    return img_final
        

if __name__ == "__main__":

    help_message = "Run like this: python main.py [path to image]"

    if len(sys.argv)<2:
        print(help_message)
    else:
        cleaned = main(sys.argv[1])
        if cleaned is not None:
            cv2.imwrite(f'clearned.png', cleaned)
        else:
            print("Something Worng. Image not found")
        