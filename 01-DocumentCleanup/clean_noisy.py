#Script feito no windows, para executar só usar Python3
import cv2
import numpy as np
import os
from scipy import ndimage, misc
import matplotlib.pyplot as plt

# ------------------------------------------Funções------------------------------------------
'''
    Função Criada para removar o ruído das imagens
'''
def clean_threshold(img, threshold_blocksize=15, threshold_c=30):
    # Converto a imagem para tons de cinza deixando apenas um canal de cor para ser manipulado
    # pelas operações posteriores:
    img = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    # Janela que passará pela imagem e realizará as algumas das operações correspondentes
    kernel = np.ones((2,2),np.uint8)
    # O primeiro passo é realizar uma limiarização na imagem com a função abaixo, isso que já retira uma grande parte do ruído
    # nessa etapa fiz uma limiarização invertida para transformar o que era preto em branco e branco em preto para facilitar as etapas
    # das transformações morfologização abaixo (que geralmente são aplicadas nesse formato)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,threshold_blocksize,threshold_c)
    # Primeiro dilatei as letras para eleminar algum ruído ao seu redor
    img = cv2.dilate(img,kernel,iterations = 1)
    # Depois realizei a operação de fechamento, essa que vai prevenir 
    # letras borradas que podem ter acontecido no passo de limiarização
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # Finalmente, aplicada essas operações, retornei a letra para sua formação original, seguindo os 
    # objetivos propostos e também para o grau de cinza anterior ao passo de limiarização
    img = cv2.erode(img,kernel,iterations = 1)
    img = 255 - img
    # Retorna-se a imagem para três canais de cores RGB:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

'''
    Função criada para realizar a operação de aumento de brilho nas imagens
'''
def bright_increase(img, m):
    # Deixando o alpha como 1, a única coisa que essa função fará é somar o m em cada pixel e caso o valor passe de
    # 255, retorna-lo para 8 bits
    return cv2.convertScaleAbs(img, alpha = 1, beta = m)

'''
    Função criada para realizar a operação de contraste nas imagens
'''
def contrast_increase(img):
    # Corvente-se a imagem para o padrão de cores LAB e depois pega-se 
    # cada um de seus canais em variáveis representativas
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Realiza-se a equalização do histograma que fará a mudança no contraste da imagem no canal L
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # Une-se novamente os canais e converte-se a imagem para RGB antes de retorna-lo ao usuário
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# ------------------------------------------Script Solicitado------------------------------------------
'''
    Fiz a separação dos resultados em duas pastas distintas, ao qual as imagens nelas por passar por uma mudança
    em seu processamento geraram resultados diferentes que melhoravam umas fotos, mas que geravam drawback em outras,
    logo caberia ver qual deles traria melhor benefício ao modelo
'''
if not os.path.isdir('./clean_data_without_contrast'):
    os.mkdir('clean_data_without_contrast')


if not os.path.isdir('./clean_data_full_preprocessing'):
    os.mkdir('clean_data_full_preprocessing')

for image_name in os.listdir('noisy_data'):
    # Aqui simplesmente carreguei a imagem e transformei ela em cinza, para 
    # que fique com apenas um canal de cor, o preto e branco
    original_image = cv2.imread('noisy_data/' + image_name)

    # Agora fiz o primeiro processo que é a remoção do ruído usando apenas limiarização e transformação morfológica
    # salvando esse resultado em seguida
    result_1 = clean_threshold(original_image)
    cv2.imwrite('clean_data_without_contrast/' + image_name, result_1)

    # Em seguida, fiz uma etapa de preparação, aumentando o brilho e contraste da imagem antes de realizar a etapa
    # da remoção de ruído, o objetivo disso é facilitar a limiarização da imagem aumentando a diferenciação entre
    # o texto e o fundo com ruído:
    img = bright_increase(original_image, 128)
    img = contrast_increase(img)
    # Aplicadas essas operações, realizei as outras de igual forma
    result = clean_threshold(img, threshold_blocksize=25, threshold_c=50)
    cv2.imwrite('clean_data_full_preprocessing/' + image_name, result)
