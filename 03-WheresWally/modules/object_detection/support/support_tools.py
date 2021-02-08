import numpy as np
import os

class supportTools():
    
    def file_path(folder):
        files = []

        for filename in os.listdir(folder):
            files.append(os.path.join(folder, filename))
            
        return files

    # --------------------------------------------------------------------------

    def get_best_bbox(detections, label_id_offset=0):
        max_idx = np.argmax(detections['detection_scores'])

        max_bb = detections['detection_boxes'][max_idx]
        max_class = detections['detection_classes'][max_idx] + label_id_offset
        max_score = detections['detection_scores'][max_idx]

        return max_score, max_class, max_bb

    # --------------------------------------------------------------------------
        
    def get_centroid_bbox(image, bbox):
        im_height, im_width, _= image.shape
        ymin, xmin, ymax, xmax = bbox

        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)

        centroid_x = int(left + ((right - left) / 2))
        centroid_y = int(top + ((bottom - top) / 2))

        centroid = (centroid_x, centroid_y)

        return centroid

    # --------------------------------------------------------------------------