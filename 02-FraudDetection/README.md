# Evaluation of Machine Learning knowledge of Felipe Lopes by Nuveo
 
Project developed for DeeperSystems in order to evaluate to domain Felipe Lopes has with Machine Learning
 
The project is intended to detect if a handwritten signature is forged, purposely altered by the user o genuine, using a Convolutional Neural Network (CNN).
 
## Getting Started
 
Due to the lack of training examples instead of training a CNN with the convolutional layer together with the fully connected layer a novel technique was used. This technique uses triplet-loss to train separately the convolutional layer of the neural network, as a result the latter will clusterize the inputs in relation to its labels or desired classes, generating what is called embeddings. After that another machine learning algorithm can be developed to learn the patterns of the generated embeddings.

 
The data was normalized by converting it to grayscale and then dividing it by 255. The performance of the models are evaluated both through loss and accuracy on training and validation sets.


### Prerequisites
 
For executing this notebook you need to have installed: tensorflow 2.x, Scikit-learn, Numpy, and Opencv 4.x, running on Python 3.x.
 
It is possible to install those within conda using, for example:
 
```
pip install tensorflow
```
 
 
## Running the tests

To run the tests it is possible to either execute test_model.py, in case that you load the pre-trained weights, fine-tune the trained models executing fine_tune_model.py or training a new embedder and classifier from scratch by executing the train_embedder.py and train_classifier.py scripts.

To execute the scripts only adjust the image paths on the scripts and execute them as follows:

```
python train_classifier.py
```

 
## Built With
 
* [Tensorflow](https://www.tensorflow.org/) - The Deep Learning framework used
* [Numpy](https://numpy.org/) - Used to some of the pre-processing and number crushing
* [Scikit-learn](https://scikit-learn.org/stable/) - Used to prepare data for the Deep Learning models
* [OpenCV](https://opencv.org/) - Used to save, load, resize pre-process the images.
 
 
## Authors
 
* **Felipe Fernandes Lopes** - *Initial work* - [FLopes](https://github.com/FelipeFLopes)
