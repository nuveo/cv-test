# Biblioteca de apredizado profundo
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *

# Biblioteca do tesseract-ocr para o python
import pytesseract as ocr

# Biblioteca para calculos
import numpy as np

# Biblioteca de visualização de imagens
import matplotlib.pyplot as plt

# Biblioteca para tratamento de dados
import pandas as pd

class Autoencoder():
    # A classe recebe como paremetro o tamanho da imagem e o otimizador
    def __init__(self, img_shape, optimizer):
        img_rows, img_columns, img_channels = img_shape
        self.img_shape = (img_rows, img_columns, 1) # Para melhorar a velocidade dos resultados, o número de canais será definido como 1
        
        
        self.autoencoder_model = self.build_model()
        self.autoencoder_model.compile(loss='mse', optimizer=optimizer)
        self.autoencoder_model.summary()
    
    # Método de criação do autoenconder
    def build_model(self):
        input_layer = Input(shape=self.img_shape)
        
        #encoder 
        x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv1')(input_layer)
        x = MaxPooling2D((2,2), padding='same', name='pool1')(x)
        x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv2')(x)
        x = MaxPooling2D((2,2), padding='same',strides=(2,1), name='pool2')(x)

        #decoder
        x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv3')(x)
        x = UpSampling2D((2,2), name='upsample1')(x)
        x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv4')(x)
        x = UpSampling2D((2,1), name='upsample2')(x)
        output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        return Model(input_layer, output_layer)
    
    # Método de treinamento
    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size, dir_save_model):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1, 
                                       mode='auto')
        history = self.autoencoder_model.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=(x_val, y_val),
                                             callbacks=[early_stopping],
                                             workers=12)
        
        # Plotando os resultados da perda ao longo do treinamento
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        print(f"Saving model on: {dir_save_model}")
        self.autoencoder_model.save(dir_save_model)
    
    # Método que fará a avaliação do modelo treinado
    def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds

# Função para carregar as imagens e retornar as imagens e seus respectivos nomes
def load_image(path, img_shape):
    img_rows, img_columns, _ = img_shape
    image_list = np.zeros((len(path), img_rows, img_columns, 1))
    image_name = []
    for i, fig in enumerate(path):
        image_name.append(fig)
        img = image.load_img(fig, color_mode='grayscale', target_size=(img_rows, img_columns))
        x = image.img_to_array(img).astype('float32')
        x = x / 255.0
        image_list[i] = x
    
    return image_list, image_name

# Função para realizar a separação dos dados para treino e validação
def train_val_split(x_train, y_train):
    rnd = np.random.RandomState(seed=42)
    perm = rnd.permutation(len(x_train))
    train_idx = perm[:int(0.8 * len(x_train))]
    val_idx = perm[int(0.8 * len(x_train)):]
    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]

# Função que realizará a extração das informações das imagens usando o tesseract-ocr
def Results(preds):
    phrases = []
    i = 0
    for pred in preds:
        pred = pred * 255
        pred = pred.reshape(336, 432)
        plt.imsave(f'denoising_data/{i}.png', pred)
        phrases.append(ocr.image_to_string(pred, lang='eng'))
        i = i + 1
    return phrases

# Plotando a diferença entre o antes e o depois de imagem ser processada
def PlotResult(imgs, preds, index):
    plt.figure(2, figsize=(15, 10))
    test = imgs[index] * 255.0
    test = test.reshape(336, 432)   
    plt.subplot(211)
    plt.imshow(test)

    pred = preds[index] * 255
    pred = pred.reshape(336, 432)
    plt.subplot(212)
    plt.imshow(pred)
    plt.show()

