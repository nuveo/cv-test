import numpy as np
import cv2
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import paths
from sklearn.metrics import accuracy_score


def extractFeatures(image):
	
	features = feature.hog(image, orientations=9,
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")

	return features

def processImage(imagePath):

	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (200, 200))

	image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	return image

def obtainDataAndLabels(path):
	
	imagePaths = list(paths.list_images(path))
	data = []
	labels = []

	for imagePath in imagePaths:
		
		label = imagePath.split(os.path.sep)[-2]

		image = processImage(imagePath)

		features = extractFeatures(image)

		data.append(features)
		labels.append(label)

	return (np.array(data), np.array(labels))

def trainModel():
	
	# Configurar os caminhos
	trainingPath = os.path.sep.join(['/root/data/signature', "training"])
	testingPath = os.path.sep.join(['/root/data/signature', "testing"])

	# Obter o dataset
	print("[INFO] Loading data...")
	(trainX, trainY) = obtainDataAndLabels(trainingPath)
	(testX, testY) = obtainDataAndLabels(testingPath)

	# Codificador de labels
	le = LabelEncoder()
	trainY = le.fit_transform(trainY)
	testY = le.transform(testY)

	# Treinar o modelo
	model = RandomForestClassifier(n_estimators=100)
	model.fit(trainX, trainY)

	# Realizar a inferência no dataset de treino
	predictions = model.predict(testX)

	# Calcular e apresentar a matriz de confusão
	confusion = confusion_matrix(testY, predictions)
	print('Confusion Matrix\n')
	print(confusion)
	cm = pd.DataFrame(confusion_matrix(testY, predictions), columns=['Diguised', 'Forged', 'Genuine'], index=['Diguised', 'Forged', 'Genuine'])
	sns.heatmap(cm, annot=True)
	plt.show()

	# Efeturar o calculo da acurácia
	print('\nAccuracy: {:.2f}\n'.format(accuracy_score(testY, predictions)))

	# Salvar o modelo
	print('[INFO] Saving model ...')
	pickle.dump(model, open('model.pkl', 'wb'))
	print('[INFO] Model saved successfully ...')

if __name__ == "__main__":
	trainModel()