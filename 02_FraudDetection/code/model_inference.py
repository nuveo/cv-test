import cv2
import pickle
import argparse

from model_training import extractFeatures, processImage


def obtainLabel(preds):

	if preds[0] == 0:
		label = 'D'
	elif preds[0] == 1:
		label = 'F'
	elif preds[0] == 2:
		label = 'G'
	else:
		return ''

	return label

def obtainColorLabel(label):

	if label == 'G':
	    color = (0, 255, 0)
	elif label == 'D':
	    color = (255, 0 , 255)
	elif label == 'F':
	    color = (0, 0, 255)
	else:
	    return ''

	return color

def makeInference():

	# Declarar argumentos
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="path to image")
	args = vars(ap.parse_args())

	# Ler o modelo
	model = pickle.load(open('model.pkl', 'rb'))

	# Ler a imagem e realizar o redimensionamento
	image = cv2.imread(args["image"])
	dim = (640, 380)
	resize = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

	# Processar imagem
	imageProcessed = processImage(args["image"])

	# Extrair features
	features = extractFeatures(imageProcessed)

	# Realizar inferẽncia
	preds = model.predict([features])

	# Obter o label referente a detecção
	label = obtainLabel(preds)

	# Obter a cor referente a detecção
	color = obtainColorLabel(label)

	# Mostrar a imagem com a classe inferida
	cv2.putText(resize, 'Classe: {}'.format(label), (500, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

	if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()

	cv2.imshow("Output", resize)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Printar a classe inferida
	print('[INFO] Classe: {}'.format(label))

if __name__ == "__main__":
	makeInference()