
"""
Object Detection From TF2 Saved Model
=====================================
"""

from itertools import count
import os
import cv2
import csv
import time
import argparse
import numpy as np
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Forces matplotlib to use correct window management in ubuntu 18.04   

def dirPath(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def getImagePaths(image_folder):

    image_paths=[]
    image_names = os.listdir(image_folder)
    for image_name in image_names:
        if image_name.endswith('.jpg'):
            image_path = image_folder+image_name
            image_paths.append(str(image_path))
    return image_paths

def loadImage(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    return image

def main():

    parser = argparse.ArgumentParser(description='Tries to find Wally in images from input folder.')
    parser.add_argument('--model_dir', type=str, default='FindWallyModel/saved_model', help='model directory')
    parser.add_argument('--input_image_dir', type=dirPath, default='TestSet/', help='image input directory')
    parser.add_argument('--save', type=bool, default=True, help='save processed images (True or False)')
    parser.add_argument('--output_image_dir', type=dirPath, default='output/', help='output image directory')
    parser.add_argument('--threshold', type=float, default=0.6, help='detection confidence threshold')

    args = parser.parse_args()

    # print('Loading model...', end='')
    # start_time = time.time()

    #* load saved model and build the detection function
    detect_fn = tf.saved_model.load(args.model_dir)

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print('Done! Took {} seconds'.format(elapsed_time))

    image_paths = getImagePaths(args.input_image_dir)

    with open(args.output_image_dir + 'wally_picture_location.csv', mode='w') as results_file:

        #* .csv file in output directory for centroid positions
        results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    

        for image_path in image_paths:
            
            image_name = image_path.rsplit('/',1)
            image_name = image_name[-1]

            print('Running inference for {}... '.format(image_path), end='')

            image = loadImage(image_path)
            height,width = image.shape[:2]
            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]
            # Process image
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
   
            counter = 0    
            for i in range(detections['num_detections']):
                #* considers detections with confidence above given threshold
                if detections['detection_scores'][i] > args.threshold:
                    
                    counter += 1
                    #* top-left and bottom-right corners of bounding box in relative coordinates
                    y1,x1,y2,x2 = detections['detection_boxes'][i]

                    #* saves images with bounding boxes if save argument is True
                    if args.save == True:

                        image_processed = image.copy()
                        image_processed = cv2.rectangle(image_processed,(int(x1*width),int(y1*height)),(int(x2*width),int(y2*height)),(0,255,0),3)
                        image_processed = cv2.cvtColor(image_processed,cv2.COLOR_RGB2BGR)
                        cv2.imwrite(args.output_image_dir+image_name,image_processed)

                    #* writes centroids to .csv file
                    results_writer.writerow([image_name, str(int((x1+x2)*width/2)), str(int((y1+y2)*height/2))])
            if counter < 1:
                results_writer.writerow([image_name, '', ''])

            print('Done!')


if __name__ == '__main__':

    main()