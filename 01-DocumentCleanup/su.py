"""
author: Peb Ruswono Aryan

Binarization Algorithm by Su et al.

@inproceedings{Su:2010:BHD:1815330.1815351,
 author = {Su, Bolan and Lu, Shijian and Tan, Chew Lim},
 title = {Binarization of Historical Document Images Using the Local Maximum and Minimum},
 booktitle = {Proceedings of the 9th IAPR International Workshop on Document Analysis Systems},
 series = {DAS '10},
 year = {2010},
 isbn = {978-1-60558-773-8},
 location = {Boston, Massachusetts, USA},
 pages = {159--166},
 numpages = {8},
 url = {http://doi.acm.org/10.1145/1815330.1815351},
 doi = {10.1145/1815330.1815351},
 acmid = {1815351},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {document image analysis, document image binarization, image contrast, image pixel classification},
} 

"""
import sys
import numpy as np 
import cv2
import os.path as path

nfns = [
        lambda x: np.roll(x, -1, axis=0), 
        lambda x: np.roll(np.roll(x, 1, axis=1), -1, axis=0),
        lambda x: np.roll(x, 1, axis=1),
        lambda x: np.roll(np.roll(x, 1, axis=1), 1, axis=0),
        lambda x: np.roll(x, 1, axis=0),
        lambda x: np.roll(np.roll(x, -1, axis=1), 1, axis=0),
        lambda x: np.roll(x, -1, axis=1),
        lambda x: np.roll(np.roll(x, -1, axis=1), -1, axis=0)
        ]

def localminmax(img, fns):
    mi = img.astype(np.float64)
    ma = img.astype(np.float64)
    for i in range(len(fns)):
        rolled = fns[i](img)
        mi = np.minimum(mi, rolled)
        ma = np.maximum(ma, rolled)
    result = (ma-mi)/(mi+ma+1e-16)
    return result

def numnb(bi, fns):
    nb = bi.astype(np.float64)
    i = np.zeros(bi.shape, nb.dtype)
    i[bi==bi.max()] = 1
    i[bi==bi.min()] = 0
    for fn in fns:
        nb += fn(i)
    return nb

def rescale(r,maxvalue=255):
    mi = r.min()
    return maxvalue*(r-mi)/(r.max()-mi)    

def binarize(img):
    gfn = nfns
    N_MIN = 4

    g = img
    I = g.astype(np.float64)

    cimg = localminmax(I, gfn)
    _, ocimg = cv2.threshold(rescale(cimg).astype(g.dtype), 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    E = ocimg.astype(np.float64)

    N_e = numnb(ocimg, gfn)
    nbmask = N_e>0

    E_mean = np.zeros(I.shape, dtype=np.float64)

    for fn in gfn:
        E_mean += fn(I)*fn(E)

    E_mean[nbmask] /= N_e[nbmask]

    E_var = np.zeros(I.shape, dtype=np.float64)
    for fn in gfn:
        tmp = (fn(I)-E_mean)*fn(E)
        E_var += tmp*tmp

    E_var[nbmask] /= N_e[nbmask]
    E_std = np.sqrt(E_var)*.5

    R = np.ones(I.shape)
    R[(I<=E_mean+E_std)&(N_e>=N_MIN)] = 0
    
    # Normalizando para CV_8U, pois este formato é pre-requisito para a etapa de rotação.
    img_n = cv2.normalize(R, R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return img_n
