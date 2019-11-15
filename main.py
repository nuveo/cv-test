from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import sys

tf.compat.v1.disable_eager_execution()
pd.set_option('max_colwidth', 300)
classes = ['diguise',"genuine","reference","simulated"]

def predict(img):
    
    model = load_model('./best_model_mobilenet_full1.h5')
    img = cv2.imread(img)
    img = preprocess_input(img)
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, axis=0)
    result = model.predict  (img)
    result = np.argmax(result)
    res = classes[result]
    return res


if __name__=="__main__":
    response = predict(sys.argv[1])
    print(response)