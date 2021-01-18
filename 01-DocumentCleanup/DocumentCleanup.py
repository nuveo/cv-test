import cv2
import numpy as np
from copy import copy
import os

# Calcular o ângulo do texto
def getSkewAngle(image):
    # Cria uma cópia e, com base nesta: 
    # Converte para escala de cinza, aplica a função GaussianBlurr (Embaçar a imagem) e binariza a mesma com base na imagem borrada.
    newImage = copy(image)
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,5)
    # Inverte os valores binários dos pixels para a aplicação da dilatação.
    thresh = cv2.bitwise_not(thresh)

    # Aplica a dilatação para transformar os textos em retângulos representando as linhas.
    # Faz uso de um Kernel com um eixo X maior para tentar preencher o máximo dos espaços entre os caracteres da mesma linha.
    # Aplica um eixo Y menor para tentar melhor separar as linhas entre os blocos de texto.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Encontra os contornos, ordena da maior para a menor e pega a maior como referência para o ângulo.
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Chama a função minAreaRect para encontrar o ângulo do maior contorno encontrado.
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determina o ângulo. Converte o ângulo para o valor que foi originalmente usado para obert a imagem distorcida.
    # A função retorna valores de ângulo no intervalo [-90, 0).
    # Conforme o retângulo gira no sentido horário, o valor do ângulo aumenta para zero.
    # Quando zero é alcançado, o ângulo é colocado novamente em -90 graus e o processo continua.
    angle = minAreaRect[-1]

    # Se o ângulo for menor que -45 graus, precisamos adicionar 90 graus ao ângulo e pegar o ângulo inverso
    if angle < -45:
        angle = 90 + angle
    # Caso contrário, apenas continua.
    return -1.0 * angle

# Rotacionar o texto em relação a centroide do seu retângulo envolvente
def rotateImage(image, angle: float):
    newImage = copy(image)
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    return newImage

# Função responsável por rodar os processor de rotação do texto
def deskew(image):
    angle = getSkewAngle(image)
    # Faço novamente a verificação da angulatura menor que -45 
    # por conta de alguns textos que não deram certo com apenas 1 verificação
    if angle < -45:
        angle = angle + 90
    return rotateImage(image, -1.0 * angle)

if __name__ == '__main__':

    # Busca o diretório atual.
    curr_path = os.getcwd()
    path = curr_path + '/treated_images'

    # Tenta criar o diretório atual.
    if os.path.isdir('./treated_images'):
        print('Diretório Treated_images já existente')
    else:
        try:
            os.mkdir('treated_images')
        except OSError:
            print ("Falha ao criar o diretório %s." % path)
        else:
            print ("Sucesso ao criar o diretório %s!" % path)

    # Busca todos os arquivos do diretório "\noisy_data".
    filelist = os.listdir("./noisy_data/")

    for file in filelist:
        # Caso o arquivo possua a extensão ".png"...
        if file.endswith(".png"):
            # Leitura da imagem.
            image = cv2.imread("./noisy_data/" + str(file))

            # Chama a função responsável pela rotação do texto.
            rotated_image = deskew(copy(image))
            # Transforma a imagem em escala de cinza.
            gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
            # Binariza a imagem em escala de cinza com um filtro adaptativo 
            # (melhor do que OTSU nesse caso, pois nosso cérebro nos engana fazendo acharmos que existe um limiar perfeito).
            thresh = cv2.adaptiveThreshold(gray ,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,30)
            
            # Tentativa de melhorar a imagem limiarizada final (Abertura e Fechamento também foram testadas).
            # kernel = np.array([[0, 0, 0],
            #                    [0, 1, 1],
            #                    [0, 1, 0]], dtype=np.uint8)
            # thresh = cv2.bitwise_not(thresh)
            # erosion = cv2.erode(thresh,kernel,iterations = 1)
            # dilation = cv2.dilate(thresh,kernel,iterations = 1)
            # thresh = cv2.bitwise_not(thresh)           
            # dilation = cv2.bitwise_not(dilation)
            # erosion = cv2.bitwise_not(erosion)

            # Escrita da imagem no diretório criado.
            cv2.imwrite(os.path.join(path , str(file)), thresh)
            
            # Caso queiram visualizar as imagens durante a execução.
            # cv2.imshow('original', image)
            # cv2.imshow('thresh', thresh)
            # cv2.imshow('erosion', erosion)
            # cv2.imshow('dilation', dilation)
            cv2.waitKey()