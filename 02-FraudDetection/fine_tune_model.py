import os

import cv2

import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa



image_width = 200
image_height = 100
embeddings_size = 256
n_classes = 4
input_shape = (1,image_height, image_width)

#### Further Training
reference_path = "TestSet/Reference/"

paths = [reference_path]

reference_image_names = os.listdir(reference_path)

names = [reference_image_names]

images = []
labels = []

for directory, type in zip(paths, names):
    for label in range(n_classes):
        for name in type:
            img = cv2.imread(directory+name)
            img = cv2.resize(img,(image_width,image_height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
            labels.append(label)


images = np.array(images)/255
labels = np.array(labels)

images = images.astype('float32')
labels = labels.astype('float32')

images = np.reshape(images, (-1,1,image_height, image_width))


embedder = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='elu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='elu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(embeddings_size, activation='linear'),
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
])



classifier = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='elu', input_shape=(embeddings_size,)),
    tf.keras.layers.Dense(8, activation='elu'),
    tf.keras.layers.Dense(n_classes, activation='softmax'),
])

embedder.load_weights('./checkpoints/best_model_embedder')
classifier.load_weights('./checkpoints/best_model_classifier')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/best_model_embedder", monitor='mse', verbose=1,
    save_best_only=True, save_weights_only=True,
    save_frequency=1)

# Compile the model
embedder.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tfa.losses.TripletSemiHardLoss(), metrics = ['mse'])

history = embedder.fit(
    images,labels,
    epochs=70, callbacks=[checkpoint_callback])



embeddings_train = embedder.predict(images)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/best_model_classifier", monitor='sparse_categorical_accuracy', verbose=1,
    save_best_only=True, save_weights_only=True,
    save_frequency=1)

# Compile the model
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy']
    )

history = classifier.fit(
    embeddings_train,labels,
    epochs=2, callbacks=[checkpoint_callback])
