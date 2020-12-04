#!/usr/bin/env python
# coding: utf-8

# Produced by @Rafael Razeira
# 
# This script classify a image or images from a path or a directory with ONLY HAS IMAGES TO CLASSIFY
# and returns a string of F for forged, G for genuine or D for disguised 
#
# usage: python3 Classify_Signatures.py -m keras_model_path -img image_path
#                       
#                                   or
#        python3 Classify_Signatures.py -m keras_model_path -dir-img directory_image_path
# ==============================================================================

# importing the libs
import numpy as np
import os
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.preprocessing import image

# function which returns a siamese model from input shape:
# this function recieves:
# input_shape: tuple, the shape of images, example (100,100,1) is a (100,100) image with grayscale
#
# return a keras instance model

def get_siamese_model(input_shape):
    """
        Model architecture
    """
    
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    initialize_weights = RandomNormal(mean=0.0, stddev=0.01, seed=42)
    
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    initialize_bias = RandomNormal(mean=0.5, stddev=0.01, seed=42)
    
    # Define the tensors for the two input images
    left_input = Input(input_shape, name = 'I1')
    right_input = Input(input_shape, name = 'I2')
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(3,activation='sigmoid',bias_initializer=initialize_bias, name='Out')(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net

# Instance and configurate the argParser
ap = argparse.ArgumentParser()

ap.add_argument("-m", 
                "--model",
                required=True,
                help="path to model keras weights .hdf5")

ap.add_argument("-img", 
                "--image",
                required=False,
                help="path to classify a single image")

ap.add_argument("-dir-img", 
                "--directory-image",
                required=False,
                help="path to classify a directory which contains only images")

args = vars(ap.parse_args())

one_shot_image_path = 'Reference/R007.png'

height, width = 140, 140

model = get_siamese_model((height, width, 1))

print('\nTrying to load the model:')

try:
    model.load_weights(args["model"])

except ValueError:   
    print('\n Erro ao carregar modelo!')

# if there is a image to classify
if (args["image"]):

    # loading the image
    img = image.load_img(args["image"], 
                                 color_mode='grayscale',
                                 target_size=(height, width))

    img = image.img_to_array(img) # convert to numpy

    img /= 255.
    img = np.expand_dims(img, axis=0)

    # loading the default image
    default_img = image.load_img(one_shot_image_path, 
                                 color_mode='grayscale',
                                 target_size=(height, width))

    default_img = image.img_to_array(default_img) # convert to numpy

    default_img /= 255.
    default_img = np.expand_dims(default_img, axis=0)

    pred = model.predict({'I1': img,'I2': default_img})

    predict_class = np.argmax(pred[0])
    print(f"\nThe class of this image is {'D' if predict_class == 0 else 'G' if predict_class == 1 else 'F'}")

if (args["directory_image"]):


    # loading the default image
    default_img = image.load_img(one_shot_image_path, 
                                 color_mode='grayscale',
                                 target_size=(height, width))

    default_img = image.img_to_array(default_img) # convert to numpy

    default_img /= 255.

    images_to_classify = []
    stack_default_image = []
    images_to_classify_paths = []

    print('\n Loading the images...')

    for img in os.listdir(args["directory_image"]):

        img_path = f'{args["directory_image"]}/{img}'

        images_to_classify_paths.append(img_path)

        # loading the image
        Loaded_image = image.load_img(img_path, 
                                     color_mode='grayscale',
                                     target_size=(height, width))

        Loaded_image = image.img_to_array(Loaded_image) # convert to numpy

        Loaded_image /= 255.

        images_to_classify.append(Loaded_image)

        stack_default_image.append(default_img)

    print('\n Making the predictions...')
    pred = model.predict([np.array(images_to_classify), np.array(stack_default_image)])

    for i, prediction in enumerate(pred):

        predict_class = np.argmax(prediction)
        print(f"\n The class of image {images_to_classify_paths[i]} is {'D' if predict_class == 0 else 'G' if predict_class == 1 else 'F'}")