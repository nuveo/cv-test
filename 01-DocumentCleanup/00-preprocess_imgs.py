# code for preprocessing the set of images

# import the necessary packages
import numpy as np
import argparse
import cv2
from imutils import paths, grab_contours
from imutils.perspective import four_point_transform
import os
import logging
logging.basicConfig(level=logging.INFO)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
help="path to noisy_data folder")
args = vars(ap.parse_args())

# initialize path for mkdir output images
logging.info("Creating output directory")
basepath = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(basepath, "preprocessed_images")
if not os.path.isdir(path):
	os.mkdir(path)

# load images in noisy_data folder
logging.info("Attempt to load images")
imagePaths = list(paths.list_images(args["path"]))
sortedImgs = sorted(imagePaths,key=lambda x: int(x[-8:].replace(".png", "")))

# loop through images
count = 1
for img in sortedImgs:

	# display information regarding the preprocessing stage
	logging.info(f'Preprocessing image {count} out of {len(imagePaths)}')
	count +=1

	# Load images turn to grayscale and apply Otsu's threshold
	cvimg = cv2.imread(img)
	gray = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)

	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 23)

	# adjust contrast
	thresh = cv2.multiply(thresh, 1.8)

	# create a kernel for the erode() function
	kernel = np.ones((1, 1), np.uint8)

	# erode() the image to bolden the text
	erode = cv2.erode(thresh, kernel, iterations=1)

	erode = cv2.bitwise_not(erode)

	# Find contours and remove small noise
	cnts = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	for c in cnts:
		area = cv2.contourArea(c)
		if area < 2:
			cv2.drawContours(erode, [c], -1, 0, -1)

	# Invert and apply slight Gaussian blur
	result = 255 - erode
	result = cv2.GaussianBlur(result, (3,3), 0)

	fpath = os.path.sep.join([path, "preprocessed_" + img[-8:]])
	cv2.imwrite(fpath, result)  
