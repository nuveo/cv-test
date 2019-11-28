import os

import cv2

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf



image_width = 400
image_height = 200
embeddings_size = 128
n_classes = 4
input_shape = (1,image_height, image_width)

reference_path = "TrainingSet/Reference/"
simulated_path = "TrainingSet/Simulated/"
genuine_path = "TrainingSet/Genuine/"
disguise_path = "TrainingSet/Disguise/"

paths = [reference_path, simulated_path, genuine_path, disguise_path]

reference_image_names = os.listdir(reference_path)
simulated_image_names = os.listdir(simulated_path)
genuine_image_names = os.listdir(genuine_path)
disguise_image_names = os.listdir(disguise_path)

names = [reference_image_names, simulated_image_names, genuine_image_names, disguise_image_names]

images = []
labels = []

for directory, type in zip(paths, names):
    for name, label in zip(type, range(n_classes)):
        print(directory+name)
        img = cv2.imread(directory+name)
        img = cv2.resize(img,(image_width,image_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
        labels.append(label)


pre_training_images = np.array(images)/255
pre_training_labels = np.array(labels)

pre_training_images = pre_training_images.astype('float32')
pre_training_labels = pre_training_labels.astype('float32')

pre_training_images = np.reshape(pre_training_images, (-1,1,image_height, image_width))



# Traine the network
X_train, X_test, y_train, y_test = train_test_split(pre_training_images, pre_training_labels,
                                                    test_size=0.2, random_state=42)


#Load the embedder
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

embedder.load_weights('./checkpoints/best_model_embedder')

x_train = embedder.predict(X_train)
x_test = embedder.predict(X_test)


classifier = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='elu', input_shape=(embeddings_size,)), # No activation on final dense layer
    tf.keras.layers.Dense(64, activation='elu'), # No activation on final dense layer
    tf.keras.layers.Dense(32, activation='elu'), # No activation on final dense layer
    tf.keras.layers.Dense(n_classes, activation='softmax'), # No activation on final dense layer
])

# Compile the model
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy']
    )


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/best_model_classifier", monitor='val_sparse_categorical_accuracy', verbose=1,
    save_best_only=True, save_weights_only=True,
    save_frequency=1)

# Train the network
history = classifier.fit(
    x_train,y_train, validation_data = (x_test, y_test),
    epochs=2, callbacks=[checkpoint_callback])


classifier.evaluate(x_test,y_test)
