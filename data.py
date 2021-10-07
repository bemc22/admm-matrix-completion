import os
import cv2
import numpy as np

def load_data(data_dir):
    filenames = os.listdir(data_dir)
    filenames = [ os.path.join(data_dir, name) for name in filenames]
    data =  [ cv2.imread(filename, cv2.IMREAD_GRAYSCALE) for filename in filenames ]
    return data

def load_gif_data(data_dir):
    filenames = os.listdir(data_dir)
    filenames = [ os.path.join(data_dir, name) for name in filenames]

    data = []
    for filename in filenames:
        gif_data = cv2.VideoCapture(filename)
        _, image = gif_data.read()
        gif_data.release()
        data.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    data = np.array(data)
    return data