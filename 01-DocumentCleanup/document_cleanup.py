import argparse
import sys
import os

import cv2 as cv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Text cleanup. Remove background and align the text.')

    parser.add_argument(
        'input_path',
        help='Path to the input images folder or a single file.')
    parser.add_argument(
        '-o', '--output-folder',
        help='Path to the output folder. [Default=results]',
        default='results')
    parser.add_argument(
        '-e', '--extension',
        help='File extension. All extensions supported by OpenCV are accepted, but only one at a time. [Default=.png]',
        default='.png')

    return parser.parse_args()
    

if __name__ == '__main__':

    # Parse command line arguments
    args = parse_args()

    # Check if inputs are ok
    if os.path.isdir(args.input_path):
        print('INFO: All files inside \"{}\" with \"{}\" extension will be processed.'.format(args.input_path, args.extension))
        filelist = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith(args.extension)]
    else:
        if os.path.isfile(args.input_path):
            if args.input_path.endswith(args.extension):
                print('INFO: Only {} will be processed.'.format(args.input_path))
                filelist = [args.input_path]
            else:
                print('ERROR: Input file \"{}\" does not have \"{}\" format. If \"{}\" is supported by OpenCV, you can use the input argument \"-e {}\".'.format(args.input_path, args.extension, os.path.splitext(args.input_path)[1], os.path.splitext(args.input_path)[1]))
                sys.exit(1)    
        else:
            print('ERROR: Input path \"{}\" does not exist.'.format(args.input_path))
            sys.exit(2)
    
    if not os.path.isdir('results'):    
        os.mkdir('results')

    
    # For each image, remove the background and adjust rotation
    num_files = len(filelist)
        
    for i in range(num_files):

        print('Processing {}, file {} of {}...'.format(filelist[i], i+1, num_files))

        # An input image must be grayscale
        img = cv.imread(filelist[i], cv.IMREAD_GRAYSCALE)


        # Although there are endless possibilities to filter the document (eg. 
        #  dilation, erosion, blur, etc.), empirical tests showed that a single
        #  adaptive threshold resulted in a better text segmentation
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 35)


        # Apply a median blur to remove 'pepper' noise
        blurred_img = cv.medianBlur(img, 7)
        
        # Invert the image to have white text and black background
        blurred_img = 255 - blurred_img

        # Find the rectangle that best encloses the text area
        # Rect contains centroid, width, height, and angle of the result
        coordinates = np.column_stack(np.where(blurred_img > 0))
        
        rect = cv.minAreaRect(coordinates)
        
        rect_points = cv.boxPoints(rect)
        rect_points = np.int32(rect_points)
        
        angle = cv.minAreaRect(coordinates)[-1]
        
        # Adjust the angle according to the complement of 90 degrees
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle = 90-angle
        else:
            angle = -angle
        
        # Swap x and y coordinates to fix the order
        rect_points[:, [0, 1]] = rect_points[:, [1, 0]]
        
        
        # Translate the text area to the center of the image
        h, w = img.shape[:2]
        c_x, c_y = (w // 2), (h // 2)
        
        centroid_x, centroid_y = int(rect[0][1]), int(rect[0][0])

        t_x = c_x - centroid_x
        t_y = c_y - centroid_y

        T = np.float32([[1, 0, t_x], [0, 1, t_y]])
        img = cv.warpAffine(img, T, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)


        # Compute the rotation matrix of the text's angle and use it to fix the
        #  the orientation        
        M = cv.getRotationMatrix2D((c_x, c_y), angle, 1.0)
        
        img = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)


        # Save the result image in "results" folder
        cv.imwrite(os.path.join(args.output_folder, os.path.basename(filelist[i])), img)
        
