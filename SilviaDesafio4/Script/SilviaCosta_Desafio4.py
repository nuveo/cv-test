# -*- coding: utf-8 -*-

'''Metodo:
    1-leitura dos arquivos para armazenar os dados:
        - entrada de dados para o treino (x)
        - rotulos de dados para o treino (y)
        - entrada dos dados para o teste
    2- Pre-processamento dos caracteres de treino e de teste
    3- Vetorizacao dos caracteres
    4- Treinamento da rede com o metodo RandomForestClassifier
    5- Obtencao dos valores de predicao
    6- Escrever os resultados no arquivo'''

#Utilizei a distribuicao Anaconda para fazer o download do python (inclui a versao py 3.8)
#Criei uma variavel de ambiente para baixar o py 3.6 pois a maioria das bibliotecas ocorreram erro no py 3.8
#Todos as bibliotecas foram baixadas através do prompt do ambiente virtual

#Importando as bibliotecas
#ML utilizando a biblioteca skelearn para aprendizagem
import numpy as np
import re
import nltk
from sklearn.datasets import load_files 
nltk.download('stopwords')
from nltk.corpus import stopwords
import csv
from sklearn.ensemble import RandomForestClassifier #Modelo de rede
from sklearn.feature_extraction.text import CountVectorizer #biblioteca para vetorizar os caracteres
from sklearn.feature_extraction.text import TfidfTransformer

def lerArquivo(diretorio, tipo):
    lista = list()
    if (tipo == 1):
        with open(diretorio) as csv_file: #Abrir o arquivo
            csv_reader = csv.reader(csv_file) #Funcao csv.reader
            lista = list() #criando uma lista para insercao dos dados lidos no arquivo
            for row in csv_reader:
                lista.append(row[0]) #insercao das linhas do arquivo na lista
        return lista
    
    if (tipo == 2): #Outro tipo pois o arquivo de teste possui algumas especificidades
        with open(diretorio) as csv_file:
            csv_reader = csv.reader(csv_file)
            lista = list()
            for row in csv_reader:
               lista.append(row)
    
        #na leitura do arquivo de teste, os caracteres usam mais de uma coluna em algumas situacoes
        #Para contornar isso, cada sub item da lista x_teste ira ser repassada para uma nova lista com as strings apenas com tamanho 1
        aux = str()
        lista2 = list()
        for i in range(0,len(lista)): 
            if len(lista[i]) != 1:
                for j in range (0, len(lista[i])):
                    aux += lista[i][j] #somando os caracteres separados na linha da string, para formar apenas 1
                lista2.append(aux) #inserindo na nova lista
                aux = str() #zerando a variavel aux
            else:
                lista2.append(lista[i])
        
        return lista2
        
def preprocessamento(lista):
    listaaux = list()
    doc = []
    for i in range(0, len(lista)):
        doc = re.sub(r'\W', ' ', str(lista[i])) #remove caracteres especiais
        doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc) #remove caracteres 'soltos' ou sozinhos
        doc = re.sub(r'\^[a-zA-Z]\s+', ' ', doc) #reitando para remover caracteres sozinhos do inicio
        doc = re.sub(r'\s+', ' ', doc, flags=re.I) #substituindo multiplos espacos
        doc = doc.lower() #convertendo para letra minuscula
        doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc) #remove caracteres 'soltos' ou sozinhos
        doc = doc.split() #separando as palavras  
        #doc1 = [stemmer.lemmatize(word) for word in doc]
        doc = ' '.join(doc)
        listaaux.append(doc) #adicionando as modificacoes na lista caract   
    return listaaux 

def treinamento(x,y,x_teste_normalizado):
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0) #definindo parametros do RandonForest
    classifier.fit(x, y) #treinando a rede
    y_pred = classifier.predict(x_teste_normalizado) #predicao com os valores de teste
    return y_pred
    

diretorio1 = "C:\\Users\\Silvia\\Documents\\SilviaDesafio4\\TrainingSet\\sms-hamspam-train.csv" #caminho para acessar o arquivo csv de treino
diretorio2 = "C:\\Users\\Silvia\\Documents\\SilviaDesafio4\\TestSet\\sms-hamspam-test.csv"  #caminho para acessar o arquivo csv de teste
x_treinoorig = lerArquivo(diretorio1,1)
x_teste = lerArquivo(diretorio2,2)

#O procedimento fez a leitura do arquivo por linha
#Entao sera necessario outro processamento nos dados de treino (x_treino) para separar os indices dos textos de mensagens
x = list() #lista para armazenar apenas as mensagens de texto
for i in range(0,len(x_treinoorig)): #Este laço percorre do indice 0 ate o tamanho da lista x_treino
    aux = x_treinoorig[i]   #variavel auxiliar para utilizar apenas uma linha da lista de treino
    x.append(aux[5:]) #inserindo atraves da funcao append os caracteres a partir do indice 5


y_rotulo = list() #lista para armazenar os rotulos das mensagens de texto: spam or ham
y = list () #lista contendo spam: 0  e ham: 1
for i in range(0,len(x_treinoorig)):
    aux = x_treinoorig[i]
    y_rotulo.append(aux[:4]) #inserir na lista y_rotulo apenas os caracteres ate o indice 4
    if y_rotulo[i] == 'spam': 
        y.append(1)  #Se a string for igual a spam, lista recebe 0
    else:
        y.append(0) #senao, lista recebe 1
#Desta maneira a variavel x contem uma lista de string com as mensagens de texto
#E a variavel y contem os rotulos contendo spam: 1  e ham: 0    

caract_treino= preprocessamento(x) #para preencher uma nova listade dados de treino com os textos pre-processados
caract_teste= preprocessamento(x_teste) #para preencher uma nova lista de dados de teste com os textos pre-processados   

#Eh necessario realizar a vetorizacao dos caracteres para valores de numeros em uma matriz
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
x_vet= vectorizer.fit_transform(caract_treino).toarray()
x_teste_vet = vectorizer.fit_transform(caract_teste).toarray()


#Para utilizar a funcao predict o conjunto de teste deve possuir a mesma dimensao na questao de colunas
#Para isso, criei uma matriz de zeros e preenchi com as informacoes da matriz x_teste
matriz = np.zeros((len(x_teste_vet),len(x_vet[0])), dtype=np.float64)    #matriz (848,1279) 

for i in range(0,len(x_teste_vet)):
    for j in range (0,len(x_teste_vet[0])):
        valor = x_teste_vet[i][j]
        matriz[i][j] = valor
x_teste_vet = matriz #renomeando

#Indicaram utilizar a uncao TfidTransformer para transformar a matriz em uma tf-idf normalizada. (tf: term frequency)
#Com ele eh possivel estabelecer pesos na matriz
tfidfconverter = TfidfTransformer()
x_normalizado= tfidfconverter.fit_transform(x_vet).toarray()
x_teste_normalizado = tfidfconverter.fit_transform(x_teste_vet).toarray()

y_predicao = treinamento(x_normalizado,y,x_teste_normalizado) #chamando a funcao de treinamento para retornar uma predicao

#Convertendo o retorno da rede para os caracteres de spam ou ham
y_rotulo_final = list()
for i in range(len(x_teste)): #recolocando os rotulos
    aux = y_predicao[i]
    if aux == 1:
        y_rotulo_final.append('spam')
    else:
        y_rotulo_final.append('ham')
  
listafinal = list()
for i in range(len(x_teste)):
    linha = str(y_rotulo_final[i]) + ' '+ str(x_teste[i])
    linha = linha.replace('[','')
    linha = linha.replace(']','')
    linha= "".join(linha)
    listafinal.append(linha)

#salvando
diretorio3 = "C:\\Users\\Silvia\\Documents\\SilviaDesafio4\\TestSet\\sms-hamspam-test-resposta.csv" #caminho para acessar o arquivo csv de treino
with open(diretorio3, 'w', newline='') as csv_file2: #Abrir o arquivo
    csv_reader = csv.reader(csv_file2) #Funcao csv.reader
    #Precisei inserir um delimitador e ficou um espacamento
    write = csv.writer(csv_file2, delimiter =' ', quotechar = ' ', quoting = csv.QUOTE_MINIMAL,  escapechar  ='')
    for linha in listafinal:
        write.writerow(linha)
csv_file2.close()
