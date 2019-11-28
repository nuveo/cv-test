import os

import cv2

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf



image_width = 200
image_height = 100
embeddings_size = 256
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

labels = tf.keras.utils.to_categorical(labels,num_classes=n_classes,dtype='float32')


# Traine the network
X_train, X_test, y_train, y_test = train_test_split(images, labels,
                                                    test_size=0.2, random_state=42)


#Load the embedder
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

embeddings_train = embedder.predict(X_train)
embeddings_test = embedder.predict(X_test)

# Compile the model
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy', metrics = ['categorical_accuracy']
    )

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/best_model_classifier", monitor='val_categorical_accuracy', verbose=1,
    save_best_only=True, save_weights_only=True,
    save_frequency=1)

# Train the network
history = classifier.fit(
    embeddings_train,y_train, validation_data = (embeddings_test, y_test),
    epochs=70, callbacks=[checkpoint_callback])


classifier.evaluate(embeddings_test,y_test)
