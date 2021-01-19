import argparse
import sys
import os

import cv2 as cv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Where\'s Wally? Look for a reference image on query images.')

    parser.add_argument(
        'reference_image',
        help='Image to be found in input path.')
    parser.add_argument(
        'input_path',
        help='Path to the input images folder or a single file.')
    parser.add_argument(
        '-e', '--extension',
        help='File extension. All extensions supported by OpenCV are accepted, but only one at a time. [Default=.jpg]',
        default='.jpg')
    parser.add_argument(
        '-o', '--output_csv',
        help='Filename of the output CSV file. [Default=output.csv]',
        default='output.csv')
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='The corners of the reference image and its centroid are shown on a new window.')
    parser.add_argument(
        '-s', '--save-results',
        action='store_true',
        help='All results are saved in \"results\" folder.')

    return parser.parse_args()


def match_image(ref_img, query_gray, sift, matcher):
    
    # Compute SIFT keypoints and descriptors of input image
    img_keypoints, img_descriptors = sift.detectAndCompute(query_gray, None)
    
    # Get top 2 matches
    matches = flann.knnMatch(ref_descriptors, img_descriptors, k=2)
    
    # Filter only the good matches
    good_matches = [m for (m, n) in matches if (m.distance < (0.7 * n.distance))]
    
    # Find inlier and outliers points
    src_points = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([img_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Identify the reference image and get the matching perspective matrix
    M, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
    
    h, w = ref_img.shape[:2]
    c_x, c_y = (w // 2), (h // 2)
    
    # Compute the corners and centroid (last point) coordinates
    final_points = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0], [c_x, c_y]]).reshape(-1, 1, 2)
    final_points = cv.perspectiveTransform(final_points, M)
    
    return final_points
    

if __name__ == '__main__':

    # Parse command line arguments
    args = parse_args()

    # Check if inputs are ok    
    if not os.path.isfile(args.reference_image):
        print('ERROR: Reference image \"{}\" does not exist.'.format(args.reference_image))
        sys.exit(1)
              
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
                sys.exit(2)    
        else:
            print('ERROR: Input path \"{}\" does not exist.'.format(args.input_path))
            sys.exit(3)
    
    if args.save_results and not os.path.isdir('results'):    
        os.mkdir('results')
    
    
    # Create output file to hold filename, centroid_x, centroid_y
    csv_file = open(args.output_csv, 'w')
        
    # SIFT detector to identify keypoints on the images
    sift = cv.SIFT_create()
    
    # FLANN matcher will match the reference image with the query image
    index_params = dict(algorithm=1, trees=5)
    search_params = {}
    
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    # Reference image preprocessing
    ref_img = cv.imread(args.reference_image)
    ref_gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    
    # Reference image SIFT keypoints and descriptors
    ref_keypoints, ref_descriptors = sift.detectAndCompute(ref_gray, None)
                     

    # For each query image, compute reference corners and centroid
    num_files = len(filelist)
        
    for i in range(num_files):

        print('Processing {}, file {} of {}...'.format(filelist[i], i+1, num_files))

        # Input image preprocessing
        query_img = cv.imread(filelist[i])
        query_gray = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)

        # Get matcher results (corners and centroid)
        result_points = match_image(ref_img, query_gray, sift, flann)

        centroid = np.int32(result_points[-1])
        corners = result_points[0:4]
        
        # Save the result in the output CSV
        print('{},{},{}'.format(filelist[i], centroid[0,0], centroid[0,1]), file=csv_file)

        # Draw the borders and centroid on input image
        result = cv.polylines(query_img, [np.int32(corners)], -1, (255, 0, 0), 3)
        result = cv.circle(result, (centroid[0,0], centroid[0,1]), 2, (0, 200, 0), 4)
    
            
        # Show the results on screen if asked
        if args.interactive:                
            print('Centroid: ({}, {})'.format(centroid[0,0], centroid[0,1]))
                    
            cv.imshow('temp', result)
            cv.waitKey()
            cv.destroyAllWindows()
        
        
        # Save the result image in "results" folder if asked
        if args.save_results:       
            cv.imwrite(os.path.join('results/', os.path.basename(filelist[i])), result)
            
    
    # Close the output file 
    csv_file.close()
