import cv2 as cv
import numpy as np

def median_filter(img_list):

    cleaned = []

    for image in img_list:

        image_name = image.split('/')[4]
        image_number = image_name.split('.')[0]

        thisImage = cv.imread(image, 0)
        medianImage = cv.medianBlur(thisImage, 9)

        final = cv.subtract(medianImage, thisImage)
        #final = cv.bitwise_not(final)

        cleaned.append(final)
    
    return cleaned


def binarize(images):

    binarized = []

    for image in images:

        image = np.double(cv.bitwise_not(image))

        image[image > 200] = 255
        image[image <= 200] = 0

        image = np.uint8(image)

        binarized.append(cv.bitwise_not(image))
    
    return binarized


def brightness_and_contrast(images, b, c):

    new_images = []

    for image in images:
        im = np.double(image.copy())
        im = im * c + b

        #Garantindo que os valores de pixel não vão passar de 255 e nem estar abaixo de 0
        im[im > 255] = 255
        im[im < 0] = 0

        im = np.uint8(im)

        new_images.append(im)
    
    return new_images

def apply_dilation(images_list):
    kernel = np.array([
        [1, 1, 1]
    ], dtype='uint8')

    closed = []

    for img in images_list:        
        close = cv.dilate(img,kernel,iterations = 1)
        closed.append(close)
    
    
    return closed


def rotation_correction(images_list):
    rot = []
    cont = 0
    
    for image in images_list:

        sift = cv.SIFT_create()
        kp = sift.detect(image,None)

        coords = []

        for i in kp:
            coords.append([i.pt[0], i.pt[1]])
        
        coords = np.array(coords, dtype='float32')

        angle = cv.minAreaRect(coords)[-1]

        if angle > 45:
	        angle = (angle - 90)
        
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(image, M, (w, h),
                                 flags=cv.INTER_CUBIC, 
                                 borderMode=cv.BORDER_REPLICATE)
        
        rot.append(rotated)
    
    return rot

def write_images(images, images_names):

    file_path = 'results/'

    if len(images) != len(images_names):
        print('There is a problem with the imageset provided! Aborting...')
        return
    
    else:
        for i in range(len(images_names)):           
            name = images_names[i].split('/')[4]
            cv.imwrite(file_path + name, cv.bitwise_not(images[i]))