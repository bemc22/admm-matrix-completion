import os
import cv2
import numpy as np

def load_data(data_dir):
    filenames = os.listdir(data_dir)
    data =  [ cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE) for filename in filenames ]
    return data
