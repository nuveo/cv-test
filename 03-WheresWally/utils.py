import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import ipyplot
import utils as u
import os
import cv2


def list_directory_files(path, extension):
    try:
        #files = []
        df = pd.DataFrame(columns = ['Path', 'File'])
        #Verifique se é um diretório
        if(os.path.isdir(path)): 
            entries = os.listdir(path)
            for entry in entries:
                
                #Verifique se o formato de extensão do arquivo termina com .jpg
                if entry.endswith(extension):
                    file = os.path.join(path, entry)
                    
                    #Verifique se é um arquivo
                    if os.path.isfile(file):
                        #name = entry.replace(extension, '')
                        new_row = {'Path':path+entry, 'File':entry }
                        df = df.append(new_row, ignore_index=True)
                        #print(entry)
                        #files.append(path+entry)
        
        return df
    
    except Exception as error:
        raise

def configure_dir(path, name_dir):
    new_dir = path+name_dir

    if os.path.exists(new_dir):
        pass
    else:
        os.makedirs(new_dir)
    return new_dir

def yolo_formatter(file, width, height, x, y):
    
    data = {
        'file': None,
        'calss_id': None,
        'x_center': None,
        'y_center': None,
        'w_distance': None,
        'h_distance': None
    }
    
    w_norm = 1./width
    h_norm = 1./height
    
    x_center = (x[0] + x[1])/2.0
    y_center = (y[0] + y[1])/2.0
    
    w_distance = (x[1] - x[0])
    h_distance = (y[1] - y[0])
    
    x_center = x_center*w_norm
    y_center = y_center*h_norm
    
    w_distance = w_distance*w_norm  
    h_distance = h_distance*h_norm
    
    data['file'] = file
    data['class_id'] = 0
    data['x_center'] = round(x_center,6)
    data['y_center'] = round(y_center,6)
    data['w_distance'] = round(w_distance,6)
    data['h_distance'] = round(h_distance,6)
    
    return data


def processing_data_json(tuple_points, PATH_TRAINING_SET, image_name):
    df_points = pd.DataFrame(list(tuple_points), columns=['x', 'y'])
                        
    xmin = df_points.x.min()
    xmax = df_points.x.max()
    ymin = df_points.y.min()
    ymax = df_points.y.max()

    #print("Xmin: {0}\nXmax: {1}\nYmin: {2}\nYmax: {3}".format(xmin, xmax, ymin, ymax))

    im=Image.open(PATH_TRAINING_SET+image_name)
    height= int(im.size[1])
    width= int(im.size[0])
    #print("Altura: {0}\nLargura: {1}".format(height, width))
    
    return xmin, xmax, ymin, ymax, height, width

def insert_centroid(img, image_name, DF_CENTROIDS):
    x = DF_CENTROIDS.loc[DF_CENTROIDS['NameFile']==image_name]['X'].item()
    y = DF_CENTROIDS.loc[DF_CENTROIDS['NameFile']==image_name]['Y'].item()
    img = cv2.circle(img, (x, y), radius=25, color=(255, 0, 0), thickness=10)
    
    return img

def extract_info_from_json(row):
    info_json = pd.read_json(row, lines=True) #lines: True Read the file as a json object per line.
    points = info_json['shapes'].to_dict()[0][0]['points']
    image_name = info_json['imagePath'][0]
    tuple_points = [tuple(point) for point in points]
    
    return info_json, points, image_name, tuple_points