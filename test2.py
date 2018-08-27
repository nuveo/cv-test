import pytesseract
from utils import *
from PIL import Image, ImageFilter
from scipy.ndimage import rotate
import argparse

DISPLAY = False

class OCR:
    """
    Find block of text in a image, improve the image quality and run tesseract on it.
    """
    def __init__(self,tess):

        pytesseract.pytesseract.tesseract_cmd = tess
        self.config = "--psm 7"

    def show(self,image):
        """
        Show image of the process to debug
        :param image: numpy array image
        """

        if DISPLAY: cv2.imshow("Debug",image)
        while 1:
            if cv2.waitKey(2) == 27 or not DISPLAY:
                break

    def crop_image(self, roi, img):
        """
        Crop a image
        :param image: numpy array image
        :return crop: Cropped image
        """
        crop = img[roi[1]:roi[3], roi[0]:roi[2]]
        return crop

    def remove_useless(self,img):
        """"" 
        Get all the regions that contain text on the input image.
        :param img: Gray scaled image in numpy array foramt
        :return regions: A list with only valid regions [[x1,y1,x2,y2],[x1,y1,x2,y2],...]

        """""

        gray = img.copy()
        th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        kernel = np.ones((5, 5), np.uint8)
        th = cv2.dilate(th, kernel, iterations=3)
        th = cv2.erode(th, kernel, iterations=4)

        contours = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        regions = []

        for c in contours:
            if cv2.contourArea(c) > 5000:
                x, y, w, h = cv2.boundingRect(c)
                regions.append([x,y,x+w,y+h])

        crop = self.crop_image(regions[0], img)
        crop = deskew(crop)
        self.show(crop)
        return crop

    def get_text_x_regions(self, img):
        """"" 
         Get all the regions that contain text on the input image on x axes.
         :param img: Gray scaled image in numpy array foramt
         :return regions: A list with only valid regions [[x1,y1,x2,y2],[x1,y1,x2,y2],...]

         """""

        gray = img.copy()
        gray = cv2.fastNlMeansDenoising(gray, None, 9, 13)
        to_show = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 91, 12)
        th2 = denoising(th, 1000)
        th = th -th2
        th = denoising(th, 100)

        kernel = np.ones((2, 3), np.uint8)
        th = cv2.dilate(th, kernel, iterations=5)

        contours = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        regions = []
        for c in contours:
            # print(cv2.contourArea(c))
            if 50000 > cv2.contourArea(c) > 1000:
                x, y, w, h = cv2.boundingRect(c)
                if not h >w and not w >10*h:
                    regions.append([x-5, y-5, x + w+5, y + h+5])
                    cv2.rectangle(to_show, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for bbox in regions:
            cv2.rectangle(to_show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        return regions

    def get_text_y_regions(self, img):
        """"" 
         Get all the regions that contain text on the input image on y axes.
         :param img: Gray scaled image in numpy array foramt
         :return regions: A list with only valid regions [[x1,y1,x2,y2],[x1,y1,x2,y2],...]

         """""

        gray = img.copy()
        gray = cv2.fastNlMeansDenoising(gray, None, 9, 13)
        to_show = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 91, 12)
        th2 = denoising(th, 1000)
        th = th - th2
        th = denoising(th, 50)

        kernel = np.ones((3, 2), np.uint8)
        th = cv2.dilate(th, kernel, iterations=8)
        contours = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        regions = []
        for c in contours:
            # print(cv2.contourArea(c))
            if 50000 > cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                if not w > h and not h > 10 * w:
                    bbox = np.array([x - 5, y - 5, x + w + 5, y + h + 5])
                    bbox[bbox < 0] =0
                    regions.append(bbox)
                    cv2.rectangle(to_show, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for bbox in regions:
            cv2.rectangle(to_show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        self.show(to_show)
        return regions

    def remove_border(self, img):
        """"" 
        Remove border from a cropped image that contain text.
        :param img: Cropped image in numpy array format
        :return crop: Image with no borders, only text

        """""

        coords = [1000, 1000, 0, 0]
        gray = img.copy()
        th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        th = denoising(th, 150)
        contours = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        for c in contours:
            if 20000 > cv2.contourArea(c) > 200:
                x, y, w, h = cv2.boundingRect(c)
                if not y == 0 and not w > h*5:
                    if coords[0] > x: coords[0] = x
                    if coords[1] > y: coords[1] = y
                    if coords[2] < x + w: coords[2] = x + w
                    if coords[3] < y + h: coords[3] = y + h
        coords = np.array((coords[0]-5, coords[1]-5, coords[2]+5, coords[3] +5), dtype=int)
        coords[coords < 0] = 0
        crop = self.crop_image(coords, img)
        if crop.shape == (0, 0):
            return img
        return cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    def predict(self,img):
        """"" 
        Run tesseract predict
        :param img: Image on numpy array format.
        :return text: Text found inside of the image.

        """""
        pil_im = Image.fromarray(img)
        text = pytesseract.image_to_string(pil_im, config=self.config)
        return text

    def routine(self, img_dir,txt_file):
        """"" 
        Routine to process the image and get text from it.
        :param img_dir: Path to the image.
        :param txt_file: Path to output txt file

        """""
        frame = Image.open(img_dir).convert("L")
        enhaced = frame.filter(ImageFilter.EDGE_ENHANCE)
        frame = np.array(enhaced)
        frame = self.remove_useless(frame)
        regions_x = self.get_text_x_regions(frame)
        regions_y = self.get_text_y_regions(frame)
        regions = regions_y +regions_x
        f = open(txt_file, "w+")
        for region in regions:
            crop = self.crop_image(region,np.array(frame))
            s = crop.shape
            if s[0] > s[1]:
                crop = rotate(crop, -90)
            s = crop.shape
            crop = cv2.resize(crop,(s[1]*2,s[0]*2),interpolation = cv2.INTER_CUBIC)
            self.show(crop)
            crop = cv2.fastNlMeansDenoising(crop, None, 9, 13)
            crop = cv2.bilateralFilter(crop, 9, 75, 75)
            crop = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 12)
            self.show(crop)
            crop = denoising(crop,100)
            crop = self.remove_border(crop)
            crop = deskew(crop)
            text = self.predict(crop)
            self.show(crop)
            if len(text) > 4:
                f.write(
                    "Text of found on: x: {} y: {} x2: {} y2 {}\n ".format(region[0], region[1], region[2], region[3]))
                f.write("\nSTART\n" + text + "\nEND\n")
        f.close()

if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="img_path", help="Image to test")
    parser.add_argument(dest="text_path", help="Path to output .txt")
    parser.add_argument(dest="tess", help="Fully path to tesseract .exe")
    args = parser.parse_args()

    ocr = OCR(args.tess)
    ocr.routine(args.img_path,args.text_path)



