"""
In this file we have functions to generate the dataset
"""

import numpy as np
import PIL.Image
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, read_csv
import pickle

class Dataset:
    """
    Process the dataset to be trained
    """

    scaler = MinMaxScaler()

    def __init__(self,FLAGS):
        """
        :param data_dir:  Path with the csv containing the columsn ["image",cx,cy] that are the correspondent label of each image
        """
        self.load_data_from_dir(FLAGS.csv_path, FLAGS.pickle_path, FLAGS.input_shape)

    def load_data_from_dir(self, data_path, pickle_path, new_image_shape):
        """
        Load all the .jpg files in the passed folder

        :param data_path: path to csv file containing the columns ['img', 'x', 'y', 'w','h']
        """

        data = read_csv(data_path, sep=',', names=['img', 'x', 'y', 'w','h'])
        image = data["img"]
        label = data[["x","y"]]
        
        #The rate of resize in each image (this is done to normalize the labels points)
        rate_x = new_image_shape[0]/data["w"]
        rate_y = new_image_shape[1]/data["h"]

        label["x"] = label["x"]*rate_x
        label["y"] = label["y"]*rate_y

        label[["x","y"]] = Dataset.scaler.fit_transform(label[["x","y"]])
        pickle.dump(Dataset.scaler, open(pickle_path, 'wb'))

        self.image_count = len(data.index)

        self.data_set = tf.data.Dataset.from_tensor_slices((image,label))
        self.data_set = self.data_set.shuffle(buffer_size=100)

    def train_test_split(self, train_split=0.9):
        """
        Split the generated dataset into train and test

        :param train_split: rate of data that will be used as training data
        :return: train and test dataset that already can be trained
        """

        self.train_ds = self.data_set.take(int(self.image_count*train_split))
        self.val_ds = self.data_set.skip(int(self.image_count*train_split))

        return self.train_ds, self.val_ds

    def configure(self, data_set, batch_size):
        """
        Some configurations to optimize the dataset

        :param data_set: dataset to be configured
        :param batch_size: size of the batch to pass to the algorithm
        :return: the transformed dataset
        """
        data_set = data_set.cache()
        data_set = data_set.batch(batch_size)
        data_set = data_set.shuffle(buffer_size=32, reshuffle_each_iteration=True)
        return data_set


def gen_image_label(image_path, label, input_shape, images_path):
    """
    This is used in a mapping function. This function will open each image to store
    as array and pass the image, label and then the dataset will be almost ready to
    train or test.

    :param image_path: image path to the image (this is passed automaticaly by map function)
    :param label: x,y points of the center in the image (this is passed automaticaly by map function)
    :param input_shape: [width,height] of the input image to the model
    :param images_path: Path to the folder where the images are
    """

    # load the raw data from the file as a string
    img = tf.io.read_file(tf.strings.join([images_path,image_path]))

    #decode image
    img = tf.io.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, input_shape)
    img = img * 1.0/255.0

    return img, label

class Predict:
    """
    This class process the input file to be predicted
    """
    def gen_image(self, image_path, input_shape):
        """
        The same map function used in gen_image_label, the difference is that we don't use a label here
        
        :param image_path: path to the image to be predicted
        :param input_shape: image shape of the input in neural network
        """

        # load the raw data from the file as a string
        img = tf.io.read_file(image_path)

        #decode image
        img = tf.io.decode_jpeg(img, channels=3)
        h_img = tf.cast(tf.shape(img)[0], tf.float64)
        w_img = tf.cast(tf.shape(img)[1],tf.float64)
        
        img = tf.image.resize(img, input_shape)
        img = img * 1.0/255.0

        self.rate_x = (w_img/input_shape[0])
        self.rate_y = (h_img/input_shape[1])

        return img

    def reverse_transform(self, point_result, pickle_path):
        """
        Reverse the transform in the point, because the point is normalized
        
        :param point_result: A tensor point with the result points
        :return: this point transformed
        """

        scaler= pickle.load(open(pickle_path, 'rb'))

        point_result = scaler.inverse_transform(point_result)
        point_result[0][0] = point_result[0][0]*self.rate_x
        point_result[0][1] =  point_result[0][1]*self.rate_y

        return point_result