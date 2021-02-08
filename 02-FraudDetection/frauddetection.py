__author__ = 'Rafael Lopes Almeida'
__email__ = 'fael.rlopes@gmail.com'
__date__ = '07/02/2021'
 
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import utils

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
PATH_IMG = './data/'
PATH_TRAIN = './data/train/'
PATH_VAL = './data/validation/'
PATH_TEST = './data/test/'
PATH_LOG = './logs/' + datetime.now().strftime('%Y%m%d-%H%M%S')


# Set parameters
# ---------------------------------------------------------------------------
IMG_NUM_TRAIN = sum(len(files) for _, _, files in os.walk(PATH_TRAIN))
IMG_NUM_TEST = sum(len(files) for _, _, files in os.walk(PATH_TEST))

BATCH_SIZE = 4
IMAGE_SIZE = (300, 300)
STEPS_PER_EPOCH = np.ceil(IMG_NUM_TRAIN/BATCH_SIZE)


# Config IMG generator
'''
Generate batches of tensor image data with real-time data augmentation.
The data will be looped over (in batches).
'''
# ---------------------------------------------------------------------------
datagen_train = ImageDataGenerator(rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1.0/255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

datagen = ImageDataGenerator(rescale=1.0/255.0)


# Batch flow from directory
'''
Takes the path to a directory and generates batches of augmented data.
'''
# ---------------------------------------------------------------------------
gen_img_train = datagen_train.flow_from_directory(
        PATH_TRAIN, 
        batch_size=BATCH_SIZE, class_mode='categorical',
        target_size=IMAGE_SIZE, shuffle=True)

gen_img_val = datagen.flow_from_directory(
        PATH_VAL,
        batch_size=BATCH_SIZE, class_mode='categorical',
        target_size=IMAGE_SIZE, shuffle=True)

gen_img_test = datagen.flow_from_directory(
        PATH_TEST, 
        batch_size=BATCH_SIZE, class_mode='categorical',
        target_size=IMAGE_SIZE, shuffle=False)


# Model
# ---------------------------------------------------------------------------
# Build model
model = utils.build_model_fix()
model.summary()

# Set callbacks
'''
Tensorboard = Provide real-time vizualization of metrics
Early Stopping = Stop training automatically when conditions met
'''
# ----------------------------
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=PATH_LOG, 
                                                        histogram_freq=10,
                                                        write_graph=False,
                                                        write_images=False)
early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)

# Keras Tuner
'''
Search the optimal set of hyperparameters for your TensorFlow program.
hyperparameter tuning.
'''
# ----------------------------
# tuner_search = RandomSearch(
#         utils.build_model, max_trials = 10, 
#         objective ='val_accuracy', project_name = 'Kerastuner', 
#         directory ='./logs/')

# tuner_search.search(
#         gen_img_train, steps_per_epoch = 25,
#         epochs = 5, verbose = 1,
#         validation_data = gen_img_val)

# model = tuner_search.get_best_models(num_models=1)[0]
# model.save('./model/weights.h5')
# model.summary()


# # Train model
# # ----------------------------
results_fit = model.fit(gen_img_train, epochs = 600,
        steps_per_epoch = STEPS_PER_EPOCH,
        validation_data = gen_img_val,
                        verbose = 1,
                        callbacks = [early_stop, tensorboard_callback])
model.save('./model/weights.h5')