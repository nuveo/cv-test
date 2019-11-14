import cv2
import sys
import numpy as np

class Process:
    
    def __init__(self):
        pass

    def denoising(self,img,min_size):
        """
        Denoise a binarized image.
        :param img; Binarized image.
        :return : Denoised image.
        """
        connectivity = 10
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        img2 = np.zeros((output.shape), np.uint8)
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255

        return img2



    def rotation(self,image):
        """
        Rotate the image and centralize
        """


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)


        thresh = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        thresh-=self.denoising(thresh,10000)

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle) 
        else:
            angle = -angle

        # rotate the image to deskew it
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


        return rotated
    def data_processing(self,img):
        """
        Make all process with pre-processing
        data
        """
        im = cv2.imread(img)
        im = self.rotation(im)
        # smooth the image with alternative closing and opening
        # with an enlarging kernel
        morphological = im.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morphological = cv2.morphologyEx(morphological, cv2.MORPH_CLOSE, kernel)
        morphological = cv2.morphologyEx(morphological, cv2.MORPH_OPEN, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # take morphological gradient
        gradient = cv2.morphologyEx(morphological, cv2.MORPH_GRADIENT, kernel)


        # split the gradient image into channels
        image = np.split(np.asarray(gradient), 3, axis=2)

        channel_height, channel_width, _ = image[0].shape

#         # apply Otsu threshold to each channel
        for i in range(0, 3):
            _, image[i] = cv2.threshold(~image[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            image[i] = np.reshape(image[i], newshape=(channel_height, channel_width, 1))

        # merge the channels
        image = np.concatenate((image[0], image[1], image[2]), axis=2)
        im = self.rotation(image)
        cv2.imshow('result',im)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERRO! Use a path that has a image")
    else:
        Process().data_processing(sys.argv[1])