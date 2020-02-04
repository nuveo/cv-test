from cleanup.clean import (median, adaptative, otsu, edges)
import os
import glob
import cv2 as cv

# Edit here to contain the path to your test images
in_dir = '/home/leo/nuveo_challenge/cv-test/data/noisy_data'


def process_imgs():
    '''
    This function applies the selected filter
    '''
    imgs = glob.glob(os.path.join(in_dir, '*.png'))
    processed = median(imgs)

    for img1, img2 in zip(imgs, processed):
        cv.imwrite(img1.replace('/noisy_data', '/processed'), img2)


def main():
    process_imgs()


if __name__ == "__main__":
    main()
