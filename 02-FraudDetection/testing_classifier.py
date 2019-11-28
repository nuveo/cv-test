import os

import cv2

import numpy as np

import tensorflow as tf



image_width = 400
image_height = 200
embeddings_size = 128
n_classes = 4
input_shape = (1,image_height, image_width)


#### Testing
questioned_path = "TestSet/Questioned/"


questioned_image_names = os.listdir(questioned_path)

images = []

for name in questioned_image_names:
    img = cv2.imread(questioned_path+name)
    img = cv2.resize(img,(image_width,image_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img)

images = np.array(images)/255
images = images.astype('float32')
images = np.reshape(images, (-1,1,image_height, image_width))





embedder = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='elu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
    #tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='elu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
    #tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='elu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='elu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(embeddings_size, activation='linear'), # No activation on final dense layer
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
])

classifier = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='elu', input_shape=(embeddings_size,)), # No activation on final dense layer
    tf.keras.layers.Dense(64, activation='elu'), # No activation on final dense layer
    tf.keras.layers.Dense(32, activation='elu'), # No activation on final dense layer
    tf.keras.layers.Dense(n_classes, activation='softmax'), # No activation on final dense layer
])


embedder.load_weights('./checkpoints/best_model_embedder')
classifier.load_weights('./checkpoints/best_model_classifier')



#Predicting on the questioned dataset
x_test = embedder.predict(images)

y_pred = classifier.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)

output = []

for predict in y_pred:
    if predict == 0 or predict == 1:
        output.append('G')
    elif predict == 2:
        output.append('F')
    else:
        output.append('D')

print(output)

print("names: {} class {} \n".format(questioned_image_names, output))
