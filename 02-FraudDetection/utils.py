import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input


from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def CreateDataFrame(data_dir):
    # As listas vazias irão guardar as informações de localização, classificação e os arquivos na pasta do dataset
    path = []
    labels = []
    files = []

    # Aqui nós iteramos em cada uma das subpastas do diretório onde se encontra o dataset
    # E Adicionaremos essas informações nas listas declaradas acima
    for folder in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir, folder)):
            path.append(os.path.join(data_dir, folder, file))
            labels.append(folder)
            files.append(file)
            
    #Criamos um dicionário baseado nas informações contidas nas listas
    data = {
        "path": path,
        "classification": labels,
        "files": files
    }

    #Com o dicionario criado, definimos uma nova variável 
    #Que conterá as informações do dicionario em formato de DataFrame da biblioteca Pandas
    return pd.DataFrame(data)

#A função PlotCategorical foi criada a fim de ter uma melhor visualização dos dados extraidos da pasta do dataset
def PlotCategorical(dataframe, x, title, x_label, y_label):
    plt.figure(figsize=(10,7))
    sns.countplot(x=x, 
                  hue=x, 
                  data=dataframe)

    plt.xlabel(x_label, 
               labelpad=14)

    plt.ylabel(y_label, 
               labelpad=14)

    plt.title(title, y=1.02)

    plt.legend(loc='center left', 
               bbox_to_anchor=(1, 0.5))
    
    plt.show()

def PlotHistoryAccuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], 
                loc='upper left', 
                bbox_to_anchor=(1, 0.5))

    plt.show()

def PlotHistoryLoss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], 
                loc='center left',
                bbox_to_anchor=(1, 0.5))
    plt.show()

def PlotTSNE(features, generator):

    tsne = TSNE(n_components=2,
                verbose=1).fit_transform(X=features)

    # Os dados gerados pelo TSNE será armazenado em um dataframe para que se tenha posteriomente uma melhor visualização

    tsne_df = pd.DataFrame(tsne, 
                        columns=[
                            'Component 1', 
                            'Component 2'
                        ])


    # Aqui é criada uma nova coluna com as classes de cada dado no dataframe
    tsne_df['Class'] = generator.classes

    # Platando a distribuição dos dados gerados pelo TSNE no eixo X e Y

    plt.figure(figsize=(10,7))
    sns.scatterplot(x="Component 1", y="Component 2", data=tsne_df, hue="Class")

    plt.xlabel("Component 1", 
            labelpad=14)

    plt.ylabel("Component 2", 
                labelpad=14)

    plt.title("T-Stochastic Neighbors Embedding", y=1.02)

    plt.legend(loc='center left', 
            bbox_to_anchor=(1, 0.5))

    plt.show()

def PlotPCA(features, generator):
    pca = PCA(n_components=2).fit_transform(X=features)

    # Os dados gerados pelo PCA serão armazenados em um dataframe para que se tenha posteriomente uma melhor visualização

    pca_df = pd.DataFrame(pca, 
                        columns=[
                            'Component 1', 
                            'Component 2'
                        ])


    # Aqui é criada uma nova coluna com as classes de cada dado no dataframe
    pca_df['Class'] = generator.classes

    # Platando a distribuição dos dados gerados pelo PCA no eixo X e Y

    plt.figure(figsize=(10,7))
    sns.scatterplot(x="Component 1", y="Component 2", data=pca_df, hue="Class")

    plt.xlabel("Component 1", 
            labelpad=14)

    plt.ylabel("Component 2", 
                labelpad=14)

    plt.title("Principal Component Analysis", y=1.02)

    plt.legend(loc='center left', 
            bbox_to_anchor=(1, 0.5))

    plt.show()

def test_prediction(directory, model):
    imgs = []
    preds = []
    for img in os.listdir(directory):
        x = load_img(directory + img,
                    grayscale=False,
                    color_mode='rgb',
                    target_size=(299, 299),
                    interpolation='nearest')
        x = img_to_array(x)

        x = preprocess_input(x)

        x = x / 255.0

        x = np.reshape(x,[1,299,299,3])

        pred = model.predict_classes(x)

        if pred == 0:
            pred = 'Disguise'
        elif pred == 1:
            pred = 'Genuine'
        else:
            pred = 'Simulated'

        imgs.append(img)
        preds.append(pred)

    data = {
            "filename": imgs,
            "prediction": preds
    }



    return pd.DataFrame(data, columns=["filename", "prediction"])