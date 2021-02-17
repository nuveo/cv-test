import cv2
import numpy as np
import glob
import math
from numpy import median
import pytesseract

class ProcessImage():

    def substractBackground(self, image, blur):

        median = cv2.medianBlur(image, blur)
        dst	= cv2.subtract(median, image)

        return dst

    def convertToGray(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray

    def threshold(self, image):

        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return thresh

    def findTextAngle(self, lines):

        angleList = []
        pos_list = []
        pos_count = 0
        neg_list = []
        neg_count = 0

        # Algoritmo para verificar a média das inclinações das retas, para
        # obtermos o ângulo  do texto. Foi identificado que nas imagens
        # do dataset as linhas que indicam a inclinação do texto estão 
        # em maior quantidade do que as outras que aparecem devido a imperfeições
        # do algoritmo de subtração do contorno. Outra observação é estão apontando para sentidos opostos,
        # desta forma possuem valores diferentes de angulação (enquanto uma é positiva outra é negativa). A solução foi implementada
        # da seguinte forma:
        # 1) Obter os angulos de todas as retas e armazenar em uma lista
        # 2) Verificar a quantidade de números positivos e negativos
        # 3) Adicionar cada número em uma lista separada de números positvos e números negativos
        # 4) A quantidade maior vai informa qual a lista que 

        for i in range(0, len(lines)):

            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            angle = math.atan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
            angle = angle * 180 / 3.14

    
            angleList.append(angle)

        for num in angleList: 
    
            if num >= 0: 
                pos_count += 1
                pos_list.append(num)
        
            else: 
                neg_count += 1
                neg_list.append(num)

        if pos_count > neg_count:
            angle = median(pos_list)
        else:
            angle = median(neg_list)

        return angle

    def processImage(self, imgPath, blur):

        img = cv2.imread(imgPath)

        imgSubs = self.substractBackground(img, blur)

        dst = cv2.Canny(imgSubs, 50, 200)
       
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        
        if not lines.any():
            print("Not detect lines ...")
            return "Error ..."
        
        if lines is not None:

            # Obter o angulo do texto
            angle = self.findTextAngle(lines)

            (h, w) = img.shape[:2]

            center = (w // 2, h // 2)

            # Aplicar a rotação para alinhar o texto na horizontal
            M = cv2.getRotationMatrix2D(center, angle + 180, 1.0)
            
            rotated = cv2.warpAffine(cv2.bitwise_not(imgSubs), M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # Aplicar operações morfológicas para melhorar o texto
            imgGray = self.convertToGray(rotated)

            imgThresh = self.threshold(imgGray)

            gray = cv2.bitwise_not(imgThresh)

            kernel = np.ones((1,1),np.uint8)
            dilation = cv2.dilate(gray,kernel,iterations = 2)

            imgThresh = self.threshold(imgGray)

            # Efetuar a leitura do texto
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(imgThresh, config=custom_config)


        # Obter a número da imagem
        test = imgPath.split('/')
        afterPeriod = test[-1]
        last = afterPeriod.split('.')

        # Salvar imagem e arquivo de texto com a leitura
        cv2.imwrite('/root/output/{}.png'.format(last[0]), imgThresh)
        text_file = open('/root/output/{}_text.txt'.format(last[0]), 'w+')
        text_file.write(str(text))

        return "Success ..."

    
if __name__ == "__main__":
    image = ProcessImage()

    filenames = glob.glob("/root/dataset/*.png")

    for file in filenames:
        print(image.processImage(file, 11))