import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

# Função que busca o maior retangulo identificado do wally
def get_maior_retangulo(retangulos):
    x, y, w, h = (0,0,0,0)
    for (x1,y1,w1,h1) in retangulos:
        if w1 + h1 > w + h:
            x, y, w, h = (x1,y1,w1,h1)
    return x, y, w, h

# Função para juntar os retangulos - essa função pega os retângulos menores que identificaram partes do wally e cria um retângulo maior que engloba esses retângulos
def join_rectangles(rectangles):
    x_values = []
    y_values = []
    x_w_values = []
    y_h_values = []
    
    for (x, y, w, h) in rectangles:
        x_values.append(x)
        y_values.append(y)
        x_w_values.append(x + w)
        y_h_values.append(y + h)
        
    return min(x_values), min(y_values), max(x_w_values), max(y_h_values)

# Função para pegar o centróide do objeto identificado
def get_centroid_pts(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    
    return cx, cy



def main():
    # Set our cascade classifier we created earlier
    CASCADE_FILE = './data/cascade.xml'

    results = []

    cascade = cv2.CascadeClassifier(CASCADE_FILE)

    for file_type in ['TestSet']:
        for img in os.listdir("./"+file_type):
            if img.split(".")[1] == "jpg": ## pegando apenas os arquivos de imagens
                img_path = "./"+file_type+"/"+img

                image = cv2.imread(img_path)
                # convert to gray scale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # dimensions
                width = int(image.shape[1])
                height = int(image.shape[0])
                
                # detecting rectangles
                rectangles = cascade.detectMultiScale(gray_image, scaleFactor=1.002, minNeighbors=18,
                  minSize=(25, 25),maxSize=(int(width*0.5), int(height*0.5)))

                cx, cy = 0,0 # centroid position

                if len(rectangles) > 0:
                    # Join multiples rectangles
                    x, y, x_plus_w, y_plus_h = join_rectangles(rectangles)
                    
                    if (x_plus_w - x) > width*0.6 or (y_plus_h - y) > height*0.6: #  If the function of joining the rectangles forms a very large rectangle, it would be better to take only the largest rectangle
                        x, y, w, h = get_maior_retangulo(rectangles)
                        cv2.rectangle(gray_image,(x,y),(x+w,y+h),(0,0,0), 10)  
                        cx, cy = get_centroid_pts(x, y, w, h)
                    else:
                        cv2.rectangle(gray_image,(x, y), (x_plus_w, y_plus_h),(0,0,0), 10)
                        cx, cy = get_centroid_pts(x, y, x_plus_w - x, y_plus_h - y)

                   

                results.append([img,cx,cy])

                # ploting images
                plt.imshow(gray_image)
                plt.show()

                #save image
                cv2.imwrite('./Results/'+img, gray_image)
                
    return results


if __name__ == '__main__':
    main()