__author__ = 'Rafael Lopes Almeida'
__email__ = 'fael.rlopes@gmail.com'
__date__ = '07/02/2021'


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# ----------------------------------------------------------------------------------- Model - Keras Tuner

def build_model(hp):
    '''
    Creates CNN model for hyperparameter tuning using keras tuner.
    '''
    model = keras.Sequential([
        keras.layers.Conv2D(
            filters = hp.Int('conv_1_filter', min_value = 16, max_value = 32, step = 16),   
            kernel_size = hp.Choice('conv_1_kernel', values = [3,5,7]),
            input_shape = (300, 300, 3),               
            activation = 'relu',
            name = 'Conv2D_1'),

        keras.layers.MaxPooling2D(pool_size=(2), name='MaxPooling_1'),

        keras.layers.Conv2D(
            filters = hp.Int('conv_2_filter', min_value = 32, max_value = 64, step = 16),
            kernel_size = hp.Choice('conv_2_kernel', values = [3,5,7]),
            activation = 'relu',
            name = 'Conv2D_2'),

        keras.layers.MaxPooling2D(pool_size=(2,2), name='MaxPooling_2'),
        keras.layers.Dropout(0.2),

        keras.layers.Conv2D(
            filters = hp.Int('conv_3_filter', min_value = 64, max_value = 128, step = 16),
            kernel_size = hp.Choice('conv_3_kernel', values = [3,5,7]),
            activation = 'relu',
            name = 'Conv2D_3'),

        keras.layers.MaxPooling2D(pool_size=(2,2), name='MaxPooling_3'),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),

        keras.layers.Dense(
            units = hp.Int('dense_1_units', min_value = 32, max_value = 256, step = 16),
            activation = 'relu',
            name = 'connected'),

        keras.layers.Dense(3, activation = 'softmax', name='output')],
        name = 'KT_model')

    model.compile(optimizer = keras.optimizers.Adam(hp.Choice('learning_rate', values = [1e-3, 1e-4, 1e-5])),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])

    return model

# ----------------------------------------------------------------------------------- Model

def build_model_fix():
    '''
    Create CNN model
    '''
    model_fix = keras.Sequential([
        keras.layers.Conv2D(filters = 32,
                            kernel_size = (3,3),
                            kernel_initializer = 'he_uniform',
                            input_shape = (300, 300, 3),
                            activation = 'relu',
                            name = 'Conv2D_1'
                            ), 

        keras.layers.Conv2D(filters = 32,
                            kernel_size = (3,3),
                            activation = 'relu',
                            kernel_initializer = 'he_uniform',
                            name = 'Conv2D_2'
                            ),

        keras.layers.MaxPooling2D(pool_size=(2,2), name='MaxPooling_1'),
        keras.layers.Dropout(0.2),

        keras.layers.Conv2D(filters = 64,
                            kernel_size = (3,3),
                            activation = 'relu',
                            kernel_initializer = 'he_uniform',
                            name = 'Conv2D_3'
                            ),

        keras.layers.Conv2D(filters = 96,
                            kernel_size = (3,3),
                            activation = 'relu',
                            kernel_initializer = 'he_uniform',
                            name = 'Conv2D_4'
                            ),

        keras.layers.MaxPooling2D(pool_size=(2,2), name='MaxPooling_2'),                
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),

        keras.layers.Dense(units = 120, activation = 'relu', name='connected'),
        keras.layers.Dense(units = 3, activation = 'softmax', name='output')],
        name = 'base_model')

    model_fix.compile(optimizer = keras.optimizers.Adam(learning_rate=0.00001), 
                        loss = 'categorical_crossentropy',
                        metrics = ['accuracy'])

    return model_fix

# ----------------------------------------------------------------------------------- 