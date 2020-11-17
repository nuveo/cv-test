'''
WARNING
This file of code does not contains any usable code in the model,
are just functions used to analyse the data

To use pay atention to the paths in the begining of the file
'''

import cv2
import os
import json
import numpy as np

base_path = '/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/AugmentedImage'
csv_path = '/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/AugmentedImage/list_images.csv'

def show(image):
    image = cv2.resize(image,None, fx=0.5, fy=0.5)
    cv2.imshow('image', image)
    while True: #press esc to exit
        a = cv2.waitKey(0)
        if a==27:
            break
        elif a == 113:
            exit(0)

def get_bb(file_path):
    '''
    :param file_path: path to json file where is the information of bounding boxes
    :return: array of points to the polygon
    '''

    bb_points = []
    with open(file_path) as f:
        json_data = json.load(f)

    for shape_n in json_data['shapes']:
        bb_points.append(shape_n["points"])

    return np.array(bb_points[0])

def get_shape(file_path):

    img = cv2.imread(file_path)
    h, w, c = img.shape

    return {"width":int(w),"height":int(h)}

def draw_bb_and_show(image, list_of_points):
    list_of_points = list_of_points.reshape((-1, 1, 2))
    image = cv2.polylines(image, [list_of_points],  
                          False, (0,0,255), 3)
    show(image)

def draw_center_and_show(image, x, y):
    image_drawed = cv2.circle(image,(x,y), 10, (0,0,255), -1)
    show(image_drawed)
  

def draw_poligons():
    csv_data = np.loadtxt(csv_path, dtype=str, delimiter=',')

    for image in csv_data:
        bb = get_bb(os.path.join(
                        base_path,
                        image[0].replace('.jpg', '.json')
                    ))

        if(bb.shape[0])>3:
            print(image[0])
            img = cv2.imread(os.path.join(base_path,image[0]))
            draw_bb_and_show(img, bb)
        else:
            print(image[0])
    
def draw_centers():
    #draw centers in all images

    csv_path = "/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/result_best_model.csv"
    base_path = "/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/TestSet"
    csv_data = np.loadtxt(csv_path, dtype=str, delimiter=',')
    
    names = csv_data[:,0]
    ar_labels = np.array(list(zip(np.array(csv_data[:,1], dtype=int),
                        np.array(csv_data[:,2], dtype=int))))


    for image_name, points in zip(names,ar_labels):

        img = cv2.imread(os.path.join(base_path,image_name))
        print(image_name)
        draw_center_and_show(img, points[0], points[1])

    # for image in csv_data:
    #     img = cv2.imread(os.path.join(base_path,image[0]))
    #     draw_center_and_show(img, int(image[2]), int(image[1]))


def update_csv():
    import pandas as pd
    csv_path = "/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/AugmentedImage/list_images.csv"
    image_path = "/home/suayder/Desktop/nuveo/candidate-data/03-WheresWally/AugmentedImage/"
    
    csv = pd.read_csv(csv_path, sep=',', names = ['img', 'px', 'py'], index_col="img")

    for i in csv.index:
        shape = get_shape(image_path+i)
        csv.at[i, "width"] = shape["width"]
        csv.at[i, "height"] = shape["height"]
    csv = csv.sort_values(by="img")
    csv.to_csv(os.path.join(image_path, 'list_images_shapes.csv'), sep=',', header=False)


if __name__ == "__main__":
    #draw_poligons()
    draw_centers()
    #update_csv()