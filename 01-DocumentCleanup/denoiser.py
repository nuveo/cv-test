import cv2
import math
import os

import numpy as np

images_path = "noisy_data/"
writing_dir = "denoised_data/"
image_names = os.listdir("noisy_data/")


images = []

for path in image_names:
    img = cv2.imread(images_path+path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (h, w) = img.shape[:2]

    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY_INV,5,20)

    denoised = cv2.fastNlMeansDenoising(th3, h=60, templateWindowSize=5)

    edges = cv2.Canny(denoised, 50, 200)

    lines = cv2.HoughLinesP(edges, 1, math.pi/180.0, 100, minLineLength=w/3, maxLineGap=10)

    angles = []

    for x1, y1, x2, y2 in lines[0]:
        #cv2.line(denoised, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)


    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(denoised, M, (w, h),
    	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    images.append(img)


images = np.array(images)


for name, image in  zip(image_names, images):
  cv2.imwrite(writing_dir+str(name)+'.png',image)
