import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import glob
import itertools

#Function used to load all images.
def getFilenames(exts):
    fnames = [glob.glob(ext) for ext in exts]
    fnames = list(itertools.chain.from_iterable(fnames))
    return fnames

#Binarize and resize all of my images
def resize_and_binarize(conjunto):
    for i in range(len(conjunto)):
        thresh = cv2.threshold(conjunto[i], 230, 255, cv2.THRESH_BINARY_INV)[1]
        #resize the image
        conjunto[i] = cv2.resize(thresh, (100, 45)) 
    return conjunto

#
folder1 = ["Disguise\*.png"]
folder2 = ["Genuine\*.png"]
folder3 = ["Reference\*.png"]
folder4 = ["Simulated\*.png"]

#Building a list with all the images
img_folder1 = getFilenames(folder1)
img_folder2 = getFilenames(folder2)
img_folder3 = getFilenames(folder3)
img_folder4 = getFilenames(folder4)
final_folder = img_folder1 + img_folder2 + img_folder3 + img_folder4

#Building the y_train set 
class1_train = [0] * len(img_folder1)
class2_train = [1] * len(img_folder2)
class3_train = [2] * len(img_folder3)
class4_train = [3] * len(img_folder4)
y_train = class1_train + class2_train + class3_train + class4_train
y_train = np.asarray(y_train)



#Adding all the images to the list x_train
x_train = []
for an_image in final_folder:
    image = cv2.imread(an_image, 0)
    x_train.append(image)

#Once I have all images in x_train
#Resize and Binarize my x_train 
# I opted to resize my images to 45x100. All the images has great quality 
# so resizing them to a lower size wouldn`t have a huge impact and I still would get a great result.
x_train = resize_and_binarize(x_train)

#I normalize my data scalling them to be between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)


#building the model
model = tf.keras.models.Sequential()

#Flat images since I'm using a neural network approach.
#Neural network does not receive multi-dimensional matrix that's why I h flattened the images  
model.add(tf.keras.layers.Flatten())

#I use the relu function as the activation function
#I opted to use relu since it's the most basic activation fuction 
# for neural network problems. The valye 4500 refers to the size of each image flatten
model.add(tf.keras.layers.Dense(4500, activation=tf.nn.relu))

#adding another layer to my neural network function.
model.add(tf.keras.layers.Dense(4500, activation=tf.nn.relu))

#adding an output layer. 
# The number 4 refers to the different types of classes I'm dealing in this problem.
#I used the  softmax function as activation function because I`m dealing with 
# a classification problem and softmax function are commonly used in these problems 
model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))

#Settings for optimizing the model. I chose Adam since it is a default parameter to NN.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#I chose 4 epochs. An epochs finished when the model run through all nodes in the network. 
#When the next ephoch starts the weights are updated.
#The main goal of a NN minimize its loss. After each ephoch I update my weights.
#I noticed that 4 epochs is enough to have a very low loss and 
# a high accuracy (evaluated only against its training data).
model.fit(x_train, y_train, epochs=4)

#The final epoch usually has loss < 00474 and accuracy about 1.000
#unfortunately the test data was not provided so I couldn`t 
# evaluate the model against its test data

