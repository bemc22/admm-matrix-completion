import os
import numpy as np
import cv2
import random


class Mask():
    
    def __init__(self, mode=None, prcnt=0.5, size=(512,512)):

        MODES = ["random", 'text', "irregular"]
        self.mask = None

        if mode not in MODES:
            text = f'mask mode not valid, allowed modes: {MODES}'
            assert False, text
            
        if mode == "random":
            self.mask = (np.random.random(size) < prcnt )*1. 
        else:

            mask_dir  = os.path.join("masks", mode)
            mask_path = os.listdir(mask_dir)
            mask_path = random.choice(mask_path)
            mask_path = os.path.join(mask_dir, mask_path)
            self.mask = self.preload_mask(mask_path)

    def __call__(self, img):
        corrupted = self.mask*img 
        return corrupted, self.mask

    def preload_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128)*1.
        return mask


def svd_th(A, th):
    """Get an approximation of A with singular values grether than th.
    """
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    s[s < th] = 0
    return np.dot(u*s, vt)


def soft_th(A, th):

    S = np.sign( A )*np.max( np.abs(A) - th, 0)
    return S


def psnr(y_true, y_pred):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mse = np.mean( np.square( y_true - y_pred) )
    value = 10*np.log10(  1 / mse )        
    return value

    