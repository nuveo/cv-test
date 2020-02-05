import pandas as pd
from sklearn.svm import LinearSVC
import re

DATASET = "../dataSet/candidate-data/04-CookingRecipeDetection/TrainingSet/cooking-recipes-train.json"
DATA_TEST = "../dataSet/candidate-data/04-CookingRecipeDetection/TestSet/cooking-recipes-test.json"


#   Classe criada para gerar um dicionário de ingredientes e mapea-los em um vetor
class Dicionario:
    def __init__ (self):
        self.dicionario = {}

    #   Adiciona Palavras ao dicionario
    #       tokens = Lista de tokens
    def addText(self, tokens):
        for i in tokens:
            if not i in self.dicionario:
                self.dicionario.update({i:1})
            else:
                token_ocurrence = self.dicionario[i]
                self.dicionario.update({i:token_ocurrence+1})

    #   Retorna o Dicionário criado
    def getDict(self):
        return self.dicionario

    #   Retorna o Vetor correspondente a frequencia dos tokens
    #   Input
    #       tokens = lista de tokens
    #   Output
    #       v = Vetor correspondente de tamanho len(self.Dicionario)
    def BoW(self,tokens):

        dicAux = {}
        for x in self.dicionario.keys():
            dicAux[x] = 0

        for token in list(tokens):
            if token in self.dicionario:
                token_ocurrence = dicAux[token]
                dicAux[token] = token_ocurrence+1

        v = [0 for x in range(len(self.dicionario))]

        for index, n in enumerate(sorted(dicAux)):

            v[index] = dicAux[n]

        return(v)



#   Abrindo o dataset
df_training = pd.read_json(DATASET)

#   Verificando as classes
categories_training = df_training['cuisine']
classes_training = df_training.cuisine.unique()
cates_training = df_training.groupby('cuisine')
print("total categories:", cates_training.ngroups)
print(cates_training.size())


#   Mapeando as classes
cozinha = dict(enumerate(df_training.cuisine.unique()))
myMap = dict(map(reversed, cozinha.items()))
df_training.cuisine = df_training.cuisine.map(myMap)

#   Criando e adicionando palavras ao dicionario
dicionario = Dicionario()

for index,i in df_training.iterrows():
    dicionario.addText(i.ingredients)

#   Gerando o vetor de treinamento
x_training = []
y_training = []
for index,i in df_training.iterrows():
    x_training.append(dicionario.BoW(i.ingredients))
    y_training.append(i.cuisine)

#   Criando e treinando o modelo de ML SVM
clf = LinearSVC()
clf.fit(x_training,y_training)


#   Abrindo o dateset de test
df_test = pd.read_json(DATA_TEST)

#   extraindo os vetores de atributos
x_test = []
for index,i in df_test.iterrows():
    x_test.append(dicionario.BoW(i.ingredients))

#   Utilizando o modelo ja treinado para a classificação
y_test = clf.predict(x_test)
y_class = [cozinha[i] for i in y_test]

print(y_test)
print(y_class)
