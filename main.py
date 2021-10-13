import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as pl
from scipy.io import loadmat

data = loadmat('prueba.mat')
im = data['im']
mask = data['mask']
y = data['y']
plt.subplot(1,3,1),plt.imshow(im)
plt.subplot(1,3,2),plt.imshow(mask)
plt.subplot(1,3,3),plt.imshow(y)
plt.show()