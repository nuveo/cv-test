__author__ = 'Rafael Lopes Almeida'
__email__ = 'fael.rlopes@gmail.com'
__date__ = '07/02/2021'
'''
Run inference on image to find objective (wally) and return csv with centroid points.
'''

import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from modules.object_detection.utils import label_map_util
from modules.object_detection.utils import visualization_utils as viz_utils
from modules.object_detection.support.support_core import supportCore
from modules.object_detection.support.support_tools import supportTools

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# TF config
# ---------------------------------------------------------------------------
tf.get_logger().setLevel('ERROR')

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Disable GPU dynamic memory allocation
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Set variables
# ---------------------------------------------------------------------------
IMG_FOLDER_PATH = './data/images/test/'
PATH_TO_SAVED_MODEL = './model/my_mobilenet/saved_model'
LABEL_PATH = './model/my_mobilenet/label_map.pbtxt'

IMAGE_PATHS = supportTools.file_path(IMG_FOLDER_PATH)
category_index = label_map_util.create_category_index_from_labelmap(LABEL_PATH,
                                                                    use_display_name=True)


# Load saved model and build the detection function
# ---------------------------------------------------------------------------
print('Loading model...')
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!)


# Run inference
# ---------------------------------------------------------------------------
for img_num, image_path in enumerate(IMAGE_PATHS):

    print('Running inference for {}... '.format(image_path), end='')

    image_np = supportCore.load_image_into_numpy_array(image_path)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run detection
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be int
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    # Get best bounding box coordinates
    _, _, max_bb = supportTools.get_best_bbox(detections)

    # Get centroid coordinates of bouding box
    centroid = supportTools.get_centroid_bbox(image_np_with_detections, max_bb)

    # Generate image
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=1,
            min_score_thresh=.50,
            agnostic_mode=False)

    # ----------------------------------
    # Save image
    fig = plt.figure(frameon=False)
    plt.axis('off')
    plt.imshow(image_np_with_detections)
    fig.savefig('./output/images/'+str(image_path[19:]), bbox_inches='tight', pad_inches=0)
    
    # ----------------------------------
    # Export results
    csv_list = [image_path[14:], centroid[0], centroid[1]]
    with open('./output/output.csv', 'a', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(csv_list)

    print('done')
myfile.close()