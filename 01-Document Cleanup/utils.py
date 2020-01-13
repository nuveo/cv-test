import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import pytesseract as ocr
import numpy as np

from glob import glob

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