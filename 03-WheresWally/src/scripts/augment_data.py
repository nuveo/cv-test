'''
This is a independent script, the objective is data augmentation 
'''
import cv2
import os
import numpy as np
import math
from random import randint, uniform
import pandas as pd

def rotate(img, angle, original_object_center):
    '''
    function to rotate image and return the rotated image and point rotated

    :param angle: angle to image be rotated, in degree
    :param original_object_center: respective centroid of the object in original image
    :return: rotated image, new x,y center coordinates
    '''
    
    img_height, img_width, _ = img.shape
    center = (img_width // 2, img_height // 2)

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)

    # Apply offset
    new_img_size = int(math.ceil(cv2.norm((img_height, img_width), normType=cv2.NORM_L2)))
    rotation_matrix[0, 2] += (new_img_size - img_width) / 2
    rotation_matrix[1, 2] += (new_img_size - img_height) / 2

    # Apply rotation to the image
    image_rotated = cv2.warpAffine(img, rotation_matrix, (new_img_size, new_img_size))

    # Apply rotation to the point
    point = np.append(original_object_center, np.array([1]))
    point = np.dot(rotation_matrix, np.transpose(point))

    return image_rotated, np.array((int(point[0]), int(point[1])))


def rotation_transform(csv_centroids,base_image_path,new_dataset_path):
    """
    Do rotations operations in all images with OpenCV.

    :param csv_centroids: path to csv with image data, csv on the format [image, center_x, center_y]
    :param base_image_path: Path to images folder
    :param new_dataset_path: Path to a folder to augmented images be saved
    """

    list_images = np.loadtxt(csv_centroids, dtype=str, delimiter=',')

    list_centroids = pd.DataFrame(columns=["img", "px", "py"])


    # go through each image and sabe two rotated images

    for image in list_images:
        img = cv2.imread(os.path.join(base_image_path,image[0]))
        
        #original image
        cv2.imwrite(os.path.join(new_dataset_path,image[0]), img)

        img, point = rotate(img, randint(10,300), image[1:].astype(np.int))

        #rotated image
        cv2.imwrite(os.path.join(new_dataset_path,image[0].replace('.jpg', '_1.jpg')), img)

        list_centroids = list_centroids.append(pd.DataFrame({
                                "img":[image[0], image[0].replace('.jpg', '_1.jpg')],
                                "px":[image[1].astype(int), point[0]],
                                "py":[image[2].astype(int), point[1]]
                            }), ignore_index=True)

    for image in list_images:
        img = cv2.imread(os.path.join(base_image_path,image[0]))

        img, point = rotate(img, randint(10,300), image[1:].astype(np.int))


        cv2.imwrite(os.path.join(new_dataset_path,image[0].replace('.jpg', '_2.jpg')), img)

        list_centroids = list_centroids.append(pd.DataFrame({
                                "img":[image[0].replace('.jpg', '_2.jpg')],
                                "px": [point[0]],
                                "py": [point[1]]
                            }), ignore_index=True)

    list_centroids = list_centroids.sort_values(by="img")
    list_centroids.to_csv(os.path.join(new_dataset_path, 'list_images.csv'), sep=',', header=False,index=False)

def brightness(csv_centroids, base_image_path, new_dataset_path):
    '''
    Random brightness transformations.
    
    :param csv_centroids: path to csv with image data, csv on the format [image, center_x, center_y]
    :param base_image_path: Path to images folder
    :param new_dataset_path: Path to a folder to augmented images be saved
    '''
    
    list_images = np.loadtxt(csv_centroids, dtype=str, delimiter=',')

    list_centroids = pd.DataFrame(columns=["img", "px", "py"])

    for image in list_images:
        img = cv2.imread(os.path.join(base_image_path,image[0]))

        img = cv2.convertScaleAbs(img, alpha=uniform(0,1), beta=uniform(0,1))

        cv2.imwrite(os.path.join(new_dataset_path,image[0].replace('.jpg', '_3.jpg')), img)

        list_centroids = list_centroids.append(pd.DataFrame({
                                "img":[image[0].replace('.jpg', '_3.jpg')],
                                "px": [int(image[1])],
                                "py": [int(image[2])]
                            }), ignore_index=True)

    aux_csv = pd.read_csv(os.path.join(new_dataset_path, 'list_images.csv'), sep=',', names = ['img', 'px', 'py'])
    list_centroids = aux_csv.append(list_centroids, ignore_index=True)
    list_centroids = list_centroids.sort_values(by="img")
    list_centroids.to_csv(os.path.join(new_dataset_path, 'list_images.csv'), sep=',', header=False,index=False)

if __name__ == "__main__":
    csv_centroids = '/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/ReferenceData/without_outlier.csv'
    base_image_path = '/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/TrainingSet'
    new_dataset_path = '/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/AugmentedImage'

    rotation_transform(csv_centroids,base_image_path,new_dataset_path)
    brightness(csv_centroids,base_image_path,new_dataset_path)