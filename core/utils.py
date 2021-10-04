import numpy as np


def mask_img(img, prcnt=0.5):

    mask = (np.random.random(img.shape) < prcnt )*1. 
    img = mask*img 
    return img, mask
    