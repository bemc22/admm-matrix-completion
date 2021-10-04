import os
import cv2
import numpy as np

def load_data(data_dir):
    filenames = os.listdir(data_dir)
    filenames = [ os.path.join(data_dir, name) for name in filenames]
    data =  [ cv2.imread(filename, cv2.IMREAD_GRAYSCALE) for filename in filenames ]
    return data
