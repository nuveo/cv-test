# -*- coding: utf-8 -*-

'''Metodologia:
    1-Leitura das imagens
    2- Pre-processamento com filtro e binarizacao
    3- Uso da biblioteca pytesseract para extrair o texto das imagens
    4- Plot do texto lido em figuras .png  
    
Eu havia tentado utilizar a biblioteca tensorflow e keras para trabalhar com as imagens e ML,
com o objetivo de treinar a rede para aprender quantos graus deveria ser rotacionado
Mas com a quantidade de imagens desse diretorio, a rede nao aprendeu o suficiente e a taxa de erro foi muito alta
'''
    

#bibliotecas
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
#import tensorflow as tf
#from tensorflow import keras
import pytesseract # Módulo para a utilização da tecnologia OCR



def preprocessamento(diretorio):
    #carregando as imagens  para fazer pre processamento
    try:
        imagem = cv2.imread(diretorio, cv2.IMREAD_GRAYSCALE) #leitura da imagem e conversao para a escala de cinza
        #alterar as cores de pixels através de um limiar (150), para diminuir tons de cinza
        for y in range(0,imagem.shape[0]):
            for x in range(0, imagem.shape[1]):
                if imagem[y,x] >= 150:
                    imagem[y,x]=255
        #cv2.imshow("Imagem modificada com limiar", imagem)
    
        #suavizar com filtro
        suave = cv2.medianBlur(imagem, 15) #filtro Blur
        #uso da funcao threshold para binarizar a imagem, indice 140
        mancha = cv2.threshold(suave, 140, 255, cv2.THRESH_BINARY)[1] #binarizacao com indice alto para identificar os pixels de mancha
        #cv2.imshow('Mancha '+ str(i), mancha)
    
        #substituindo os pixels identificados como manchas, aplicando a cor branca na imagem original
        for y in range(0,imagem.shape[0]):
            for x in range(0, imagem.shape[1]):
                if mancha[y,x] == 0:
                    imagem[y,x]=255
        #Assim foram identificados os pixels de mancha e "retirados" da imagem original
            
        #binarizando e salvando
        imagem = cv2.threshold(imagem, 140, 255, cv2.THRESH_BINARY)[1]
        #cv2.imshow('imagem' +str(i), imagem)
        #Utilizei o PC no login Renato
        caminhosaida = "C:\\Users\\Renato\\Documents\\SilviaCosta\\SilviaDesafio1\\imagens_preprocessamento\\"  + str(i) + ".png"
        cv2.imwrite(caminhosaida, imagem)
    except:
        print("Imagem não encontrada ", i)
    
 
def extraindoTexto(diretorio):
    try:
        imagem= cv2.imread(diretorio, cv2.IMREAD_GRAYSCALE)        
        texto= pytesseract.image_to_string(imagem)
        texto = texto.split()
        return texto
    except:
        return ''
    
def fundoImagem(diretorio):
    try:
        imagem= cv2.imread(diretorio, cv2.IMREAD_GRAYSCALE)   
        for y in range(0,imagem.shape[0]):
            for x in range(0, imagem.shape[1]):
                imagem[y,x]= 255
        return imagem, imagem.shape
    except:
        return '',''

def textoConcatenado(texto):
    try:
        #A funcao de escrita em uma imagem nao quebra as linhas, entao o codigo a seguir irá fazer isso
        ind1 = 0 #indices para laço
        ind2 = 10
        lista = list()
        aux = str()
        tam = int(len(texto)/10) #agrupando 10 strings que foram separadas pela funcao split()
        rest = int(len(texto) % 10)
        for i in range(tam):
            aux = str()
            for i in range(ind1, ind2):
                aux += texto[i]
                aux += ' '
            lista.append(aux) #acrescendo na lista as strings concatenadas
            ind1 +=10
            ind2 +=10
        
        if rest != 0:
            aux = str()
            for i in range(ind2, len(texto)):
                aux += texto[i]
            lista.append(aux) #acrescendo na lista as strings concatenadas
        return lista
    except:
        return ''

def inserirTexto(imagem, lista, imagem_shape,i): #Funcao para inserir o texto e salvar
    try:
        y = 70
        font = cv2.FONT_HERSHEY_DUPLEX #parametros de formatacao do texto
        font_size = 0.5
        font_thickness = 1
        #j = 0
        for line in lista:
            textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
            x = int((imagem_shape[1] - textsize[0]) / 2) #centralizar
            #funcao para inserir o texto
            cv2.putText(imagem, line, (x,y), font,
                            font_size, 
                            (0,0,0), 
                            font_thickness, 
                            lineType = cv2.LINE_AA)
            #j +=1
            y +=15
        caminhosaida = "C:\\Users\\Renato\\Documents\\SilviaCosta\\SilviaDesafio1\\texto_extraido\\" + str(i) + ".png"
        cv2.imwrite(caminhosaida, imagem)     
    except:
        pass
  

for i in range (0,217):
    #Utilizei o PC no login Renato
    diretorio = "C:\\Users\\Renato\\Documents\\SilviaCosta\\SilviaDesafio1\\imagens_originais\\"  + str(i) + ".png"
    preprocessamento(diretorio) #pre-processamento das imagens
    diretorio2 = "C:\\Users\\Renato\\Documents\\SilviaCosta\\SilviaDesafio1\\imagens_preprocessamento\\" + str(i) + ".png"
    texto = extraindoTexto(diretorio2) #extraindo o texto
    #diretorio3 = "C:\\Users\\Renato\\Documents\\SilviaDesafio1\\redeneural-partefinal2\\imagens_preprocessamento\\" + str(i) + ".png"
    imagem, imagem_shape = fundoImagem(diretorio2) #Fundo branco para inserir o texto
    lista = textoConcatenado(texto) #Concatenacao das strings
    inserirTexto(imagem,lista,imagem_shape,i) #Inserindo o texto extraido e salvando
    
#Ademais, estou pesquisando sobre reconhecimento de contorno de texto ou identificacao das bordas para
#melhorar a questao da rotacao das imagens, o que facilitaria para a a biblioteca pytesseract extrair os textos
#ois devido a rotacao de algumas imagens, nao foi possivel extrair a informacao
