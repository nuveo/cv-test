import numpy as np
import cv2
import scipy.ndimage as nd
import math
import itertools

# from skimage.util import img_as_ubyte
from skimage.morphology import disk

from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola, rank, threshold_local
from su_binarization import su_binarize
# from howe import howe_binarize

class BinarizationMethod:
    def __init__(self, img_path):
        #self.img = img_as_ubyte(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # self.img_howe_path = howe_path
        self.binary = np.zeros_like(self.img)
        self.window_size = 35
        #self.img_path = ""

    def otsu_global(self):
        thresh = threshold_otsu(self.img)
        self.binary = (self.img > thresh).astype(float)

    def otsu_local(self):
        radius = 25
        selem = disk(radius)
        thresh = rank.otsu(self.img, selem)
        self.binary = (self.img > thresh).astype(float)

    def adaptative_local(self):
        thresh = threshold_local(self.img, self.window_size, offset=10)
        self.binary = (self.img > thresh).astype(float)

    def niblack_local(self):
        k = 0.8
        thresh = threshold_niblack(self.img, window_size=self.window_size, k=k)
        self.binary = (self.img > thresh).astype(float)

    def sauvola_local(self):
        k = 0.2
        thresh = threshold_sauvola(self.img, window_size=self.window_size, k=k)
        self.binary = (self.img > thresh).astype(float)

    def su_hybrid(self):
        self.binary = su_binarize(self.img)

    # def howe_hybrid(self):
        # self.binary = howe_binarize(self.img)
        #self.img_path = img_path
        # self.img = img_as_ubyte(cv2.imread(self.img_howe_path, cv2.IMREAD_GRAYSCALE))
        # Olho aqui oh! pois o texto nesse caso tem valor 1, oposto aos demais metodos
        # self.binary = np.where(self.img < 128,1.0,0.0)

    def relative_darkness(self, window_size=11, threshold_ltp=10):
        w_s = window_size
        threshold = threshold_ltp
        # find number of pixels at least $threshold less than the center value
        def below_thresh(vals):
            center_val = vals[math.ceil(vals.shape[0]/2)]
            lower_thresh = center_val - threshold
            return (vals < lower_thresh).sum()

        # find number of pixels at least $threshold greater than the center value
        def above_thresh(vals):
            center_val = vals[math.ceil(vals.shape[0]/2)]
            above_thresh = center_val + threshold
            return (vals > above_thresh).sum()

        # apply the above function convolutionally
        lower = nd.generic_filter(self.img, below_thresh, size=w_s, mode='reflect')
        upper = nd.generic_filter(self.img, above_thresh, size=w_s, mode='reflect')

        # number of values within $threshold of the center value is the remainder
        # constraint: lower + middle + upper = window_size ** 2
        middle = np.empty_like(lower)
        middle.fill(w_s*w_s)
        middle = middle - (lower + upper)

        # scale to range [0-255]
        lower = np.round(lower * (255 / (w_s * w_s)),2)
        middle = np.round(middle * (255 / (w_s * w_s)),2)
        upper = np.round(upper * (255 / (w_s * w_s)),2)

        return lower, middle, upper
