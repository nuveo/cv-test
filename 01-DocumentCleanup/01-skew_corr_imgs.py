# code for perform skew correction

# import the necessary packages
import numpy as np
import argparse
import cv2
from imutils import paths
from scipy.ndimage import interpolation as inter
import os
import logging
logging.basicConfig(level=logging.INFO)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to preprocessed_images folder")
args = vars(ap.parse_args())

# initialize path for mkdir output images
logging.info("Creating output directory")
basepath = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(basepath, "warped_images")
if not os.path.isdir(path):
	os.mkdir(path)

# load images in noisy_data folder
logging.info("Attempt to load images")
imagePaths = list(paths.list_images(args["path"]))
sortedImgs = sorted(imagePaths,key=lambda x: int(x[-8:].replace(".png", "")))

def correct_skew(image, delta=1, limit=13):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 23)

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated


# loop through images
count = 1
for img in sortedImgs:

    # display information regarding the preprocessing stage
    logging.info(f'Processing image {count-1} out of {len(imagePaths)}')
    count += 1
  
    # Load images turn to grayscale and apply Otsu's threshold
    cvimg = cv2.imread(img)
    angle, rotated = correct_skew(cvimg)
    fpath = os.path.sep.join([path, "warped_" + img[-8:]])
    cv2.imwrite(fpath, rotated)  
 