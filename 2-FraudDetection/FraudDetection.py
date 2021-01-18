import tensorflow.compat.v1 as tf
import keras
from keras import layers
from keras import Sequential, models
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import numpy as np

'''

Algoritmo para teste Nuveo - Classificação de assinaturas;

Objetivo: Classificar as assinaturas.

Observações: Algumas possíveis melhorias, como treinar com imagens com qualidade maior, não puderam ser feitas.
As imagens da nova pasta "Test" foram criadas manualmente, com base nas originais, a fim de testar o algoritmo treinado. (Para utilizar todas as imagens possíveis para o treinamento)
As imagens dentro de "Reference" foram treinadas para serem classificadas como "G - Genuine".

Escrito por: João Victor Ribeiro de Jesus

'''

# Função responsável por padronizar o tamanho das imagens de acordo com o tamanho especificado
def resizeImage(image, image_size):

    # Busca o valor do maior eixo da imagem e salva na variável "aux_size" auxiliar
    if image.shape[0] > image.shape[1]:
        aux_size = image.shape[0]
    else:
        aux_size = image.shape[1]
        
    # Cria uma nova imagem quadrada, com pixels brancos, de acordo com o valor do maior eixo encontrado
    new_image = np.zeros((aux_size,aux_size,3), np.uint8)           
    new_image[:,:,:] = 255

    # Centraliza a imagem antiga na nova imagem criada, caso o eixo X ou o eixo Y da imagem antiga seja menor (caso ela não seja quadrada)
    # metadeColuna = (tamanho da coluna da nova imagem - tamanho da coluna da imagem antiga) / 2
    # metadeColuna = (tamanho da linha da nova imagem - tamanho da linha da imagem antiga) / 2
    halfC = (new_image.shape[0] - image.shape[0]) // 2
    halfR = (new_image.shape[1] - image.shape[1]) // 2
    new_image[halfC:halfC+image.shape[0] , halfR:halfR+image.shape[1], : ] = image

    # Atribui a imagem antiga a nova imagem
    image = cv2.resize(new_image, (image_size,image_size))
    return image

# Função responsável pela população inicial dos dados nos respectivos arrays
def populateImages(train_images, train_labels, test_images, test_labels, test_images_names, genuine_images, simulated_images, disguised_images, image_size):

    # Percorre cada diretório dentro da pasta 'TrainingSet'
    for folder in os.listdir('./TrainingSet/'):
        # Percorre cada arquivo dentro de cada pasta de 'TrainiSet'
        for filename in tqdm(os.listdir('./TrainingSet/'+folder)):
            # Para cada arquivo terminado com '.png', em cada pasta com exceção da "Test"...
            if filename.endswith(".png") and folder != 'Test':
                # Faz a leitura da imagem
                image = cv2.imread(f'./TrainingSet/{folder}/{filename}')

                # Redimensiona a imagem original
                image = resizeImage(image, image_size)

                # Pre-processa a imagem para facilitar o treinamento e classificação posterior
                # Transforma em cinza, limiariza utilizando a função OTSU, inverte novamente a imagem para RGB
                # E por fim, todos os pixels pertencentes à marca da caneta recebem o valor 255
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                image = cv2.bitwise_not(image)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)                                                                                                                                                                                                                                                         

                # Normaliza os valores dos pixels para ser de 0 à 1
                image = image / 255.0 
                # Insere a imagem no array 'train_images'
                train_images.append(image)

                # Insere a classe no array 'train_labels' de acordo com a classificação
                # A classificação é obtida de acordo com a primeira letra da imagem
                if filename[:1] in 'GR':
                    genuine_images.append(image)
                    train_labels.append(0)
                elif filename[:1] == 'D':
                    disguised_images.append(image)
                    train_labels.append(1)
                else:
                    simulated_images.append(image)
                    train_labels.append(1)
            
            # Caso "folder" se chame "Test" e o arquivo termine em '.png'...
            elif filename.endswith(".png") and folder == 'Test':
                # O mesmo processo de cima, porém com as imagens para validação da acurácia
                image = cv2.imread(f'./TrainingSet/{folder}/{filename}') 
                image = resizeImage(image, image_size)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                image = cv2.bitwise_not(image)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = image / 255.0
                test_images.append(image)
                test_images_names.append(filename)
                if filename[:1] in 'GR':
                    test_labels.append(0)
                elif filename[:1] == 'D':
                    test_labels.append(1)
                else:
                    test_labels.append(2)

def increaseData(train_images, train_labels, genuine_images, simulated_images, disguised_images):
    # Utilização da função pronta do Keras para gerar novas imagens a partir do dataset de treinamento, com rotações diferentes
    data_augmentation = keras.Sequential([layers.experimental.preprocessing.RandomRotation(0.1),])

    # Popula os arrays com as imagens para o treinamento com as imagens criadas
    genuine_images = np.array(genuine_images, dtype=np.float32)
    genuine_generated_images = data_augmentation(genuine_images)
    for image in genuine_generated_images:
        train_images.append(image)
        train_labels.append(0)

    disguised_images = np.array(disguised_images, dtype=np.float32)
    disguised_generated_images = data_augmentation(disguised_images)
    for image in disguised_generated_images:
        train_images.append(image)
        train_labels.append(1)

    simulated_images = np.array(simulated_images, dtype=np.float32)
    simulated_generated_images = data_augmentation(simulated_images)
    for image in simulated_generated_images:
        train_images.append(image)
        train_labels.append(2)

def saveModel(model):

    # model.save('./model')

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Modelo da CNN salvo!")

# Função responsável pelo treinamento do modelo
def trainModel(train_images, train_labels, test_images, test_labels):

    # Não pude usar 64 bit pois o processador e a memória não permitiram
    model = models.Sequential()

    # As convulações utilizarão um kernel 3x3, 32 canais e um stride 1x1 (não diminuindo a imagem) 
    # As ações de pooling utilizarão um stride de 2x2 (diminuindo a imagem em 50%)
    model.add(layers.Conv2D(kernel_size=(5,5), filters=32, strides=(2,2), padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(kernel_size=(3,3), filters=32, strides=(1,1), padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(kernel_size=(3,3), filters=32, strides=(1,1), padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(kernel_size=(3,3), filters=32, strides=(1,1), padding='SAME', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3))

    # Para a função de perda foi utilizada a SparseCategorialCrossentropy por queremos categorizar as imagens em 3 classes em formato Integer
    model.compile(optimizer='sgd',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics=['accuracy'])              

    # Treina o modelo com os dados de treinamento e avaliando, para cada época, a acurácia
    history = model.fit(train_images, train_labels, epochs=7, 
                        validation_data=(test_images, test_labels))

    # Apresenta o histórico do treinamento
    # print(history)

    # Apresenta a atual acurácia em relação às imagens de teste
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("\nTest Accuracy:",test_acc)

    # Salvando modelo no diretório atual
    saveModel(model)

    return model

def main():
    # Definição do tamanho padronizado da imagem de entrada do modelo    
    # Não pude usar uma imagem maior pois o processador e a memória do meu computador não permitiram (i3, 8GB)
    IMG_SIZE = 600

    # Criação dos arrays vazios utilizados para popular o modelo
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    test_images_names = []

    genuine_images = []
    simulated_images = []
    disguised_images = []

    # Função responsável por popular os arrays com as imagens de treinamento e de teste, de acordo com as pastas no diretório
    populateImages(train_images, train_labels, test_images, test_labels, test_images_names, genuine_images, simulated_images, disguised_images, IMG_SIZE)

    # Função responsável por criar imagens novas a partir do dataset atual, para auxiliar o treinamento da rede convulacional
    increaseData(train_images, train_labels, genuine_images, simulated_images, disguised_images)

    # Padronizando dados para entrada no modelo
    train_images = np.array(train_images,dtype=np.float32)
    test_images = np.array(test_images,dtype=np.float32)

    test_labels = np.array(test_labels)
    train_labels = np.array(train_labels)

    # Keras espera esse formato para as train_labels em formato string
    # train_labels = tf.one_hot(train_labels, 3)
    # test_labels = tf.one_hot(test_labels, 3)

    # Escreve no terminal a quantidade de imagens que temos para cada set de imagens
    print("\nTrain images:",len(train_labels))
    print("Test images:",len(test_labels), "\n")
    
    # Função responsável pelo treinamento do modelo
    model = trainModel(train_images, train_labels, test_images, test_labels)

    # Carrega modelo já treinado
    # model = models.load_model('model.h5')
    # print(model.summary())

    # Representação das classes: 0 = 'genuine', 1= 'diguised' e 2 = 'forged'
    class_names = ['G - Genuine', 'D - Diguised', 'F - Forged']
    
    predictions = model.predict(test_images)
    score = tf.nn.softmax(predictions[0]) # primeira imagem
    print("A imagem {} imagem parece pertencer à {} com a porcentagem {:.2f} de confiança.".format(test_images_names[0] ,class_names[np.argmax(score)], 100 * np.max(score))
    )


if __name__ == '__main__':
    main()




