import pandas as pd
import gensim
from gensim import corpora
from nltk.tokenize import TweetTokenizer
from gensim.models import TfidfModel
from sklearn.svm import LinearSVC
from sklearn import svm
import numpy as np
from sklearn.datasets import load_svmlight_file
import re


#   Dados treinamento
DATASET = "../dataSet/candidate-data/03-SMSSpamDetection/TrainingSet/sms-hamspam-train.csv"

#   Dados test
DATA_TEST = "../dataSet/candidate-data/03-SMSSpamDetection/TestSet/sms-hamspam-test.csv"

#   Aqui utilizamos o TweetTokenizer pois como são textos informais que possuem
#   caracteres como :D e indicado utiliza-o
tknzr = TweetTokenizer()

#   Função de tratamento de texto onde passamos todos os caracteres para minusculo
#   transformamos url e emails em "URL"
#   e números em "NUM"
def tratamentoTexto(text):
    text = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', 'URL', text.lower())
    text = re.sub(r'[0-9]+([\,|\.][0-9]+)?', "NUM", text)

    return(text)

#   Abrindo o arquivo do dataset de treinamento
df_training = pd.read_csv(DATASET, delimiter='\t', header = None)
df_training.columns = ["category","text"]
df_training = df_training[['category', 'text']]


#   Verificando as Categorias
categories_training = df_training['category']
classes_training = df_training.category.unique()
cates_training = df_training.groupby('category')
print("total categories:", cates_training.ngroups)
print(cates_training.size())

#   Transformando as classes em binario
#   0 = ham
#   1 = spam
df_training.category = df_training.category.map(lambda x: 1 if x == "spam" else 0)


#   Transformamos os textos em vetores utilizando o dictionay e o tf-idf do gensim

y_training = []
x_training = []
textos = []

for index, x in df_training.iterrows():
    y_training.append(x.category)
    textos.append(tknzr.tokenize(tratamentoTexto(x.text)))

dicionario = corpora.Dictionary(textos)
dicionario.compactify()

corpus = [dicionario.doc2bow(line) for line in textos]
model = TfidfModel(corpus)

for i in textos:
    x_training.append(model[dicionario.doc2bow(i)])

#   Como o gensim retorna um vetor no modelo chamado BOW format devemos converte-lo
#   para uma matriz que o sklearn consiga processsar, o que e feito abaixo
x_training = gensim.matutils.corpus2csc(corpus=x_training,num_terms=len(dicionario))
x_training = x_training.transpose()

#   Criando e treinando o modelo de ML SVM
clf = svm.LinearSVC()
clf.fit(x_training,y_training)

#   Abrindo o dataset de teste
df_test = pd.read_csv(DATA_TEST, delimiter='\t', header = None)
df_test.columns = ["text"]
df_test = df_test[["text"]]


#   Transformamos os textos em vetores utilizando o dictionay e o tf-idf do gensim
textos = []
for index,i in df_test.iterrows():
    textos.append(tknzr.tokenize(tratamentoTexto(i.text)))
x_test = []
for i in textos:
    x_test.append(model[dicionario.doc2bow(i)])
x_test = gensim.matutils.corpus2csc(corpus=x_test,num_terms=len(dicionario))
x_test = x_test.transpose()
y_test = clf.predict(x_test)

print(y_test)
print(["ham" if y == 0 else "spam" for y in y_test])
