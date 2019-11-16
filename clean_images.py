import cv2
import numpy as np
import glob
import itertools

#Function used to load all images.
def getFilenames(exts):
    fnames = [glob.glob(ext) for ext in exts]
    fnames = list(itertools.chain.from_iterable(fnames))
    return fnames

def rotate_image(image):
	#convert the image to grayscale.
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)

	# Binarize the image etting all foreground pixels to
	# 255 and all background pixels to 0
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# Get the pixels that are greater than 0
	coords = np.column_stack(np.where(thresh > 0))
	angle = cv2.minAreaRect(coords)[-1]
	
	# rotate the image in order to make it horizontal
	if angle < -45:
		angle = -(90 + angle)
	else:
		angle = -angle

	# Now that I have the angle I should rotate, I perform the rotation.
	#I find the center of the picture and rotate it.
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h),
		flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	
	return rotated


folder1 = ["noisy_data\*.png"]
img_folder1 = getFilenames(folder1)

#I add all my images to all_images list
all_images = []
for an_image in img_folder1:
    image = cv2.imread(an_image)
    all_images.append(image)

#I rotate each image
#I use `cv2.threshold` in order to get only the text
#I convert my image to white background and  black text characters
#I save my images to the newfolder called `cleaned_images`
for i in range(len(all_images)):
	image_rotated = rotate_image(all_images[i])
	thresh = cv2.threshold(image_rotated, 125, 255, cv2.THRESH_BINARY_INV)[1]
	bi_black_white = 255 - thresh
	new_str = "cleaned_images\\" + str(i) + ".png"
	cv2.imwrite(new_str,bi_black_white)
	print(i)
