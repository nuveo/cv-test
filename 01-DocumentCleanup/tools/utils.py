import cv2
import os
import numpy as np

class Utils():

    def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):

            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)

        return images

    # --------------------------------------------

    def file_names(folder):
        files = []
        for filename in os.listdir(folder):
            files.append(filename)

        return files

    # --------------------------------------------  

    def text_conversion(sentences):

        suport_list = []
        for current_sentence in sentences:
            suport_string = str()

            for current_char in current_sentence:
                if current_char.isalpha() or current_char == ' ':
                    suport_string = suport_string + current_char

            if len(suport_string) > 0:
                suport_list.append(suport_string)

        text = []
        for list_item in suport_list:
            try:
                while list_item[0] == ' ':
                    list_item = list_item[:0] + list_item[1:]
                
                while list_item[len(list_item)-1] == ' ':
                    list_item = list_item[:len(list_item)-1]
            except:
                pass

            text.append(list_item)

        return text


    def print_text(text, fontScale=0.7, thickness=1, 
                    org_x=50, org_y=50, step=25,
                    imgx=640, imgy=480):

        font = cv2.FONT_HERSHEY_SIMPLEX 
        color = (0, 0, 0) 
        img = np.zeros([imgy, imgx, 3], dtype=np.uint8)
        img.fill(255)

        for sentences in text:
            img = cv2.putText(img, sentences, (org_x, org_y), font, 
                                fontScale, color, thickness, cv2.LINE_AA)
            org_y =  org_y + step

        return img