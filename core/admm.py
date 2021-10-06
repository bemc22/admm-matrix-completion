import numpy as np
import bm3d
import matplotlib.pyplot as plt

from core.utils import svd_th, soft_th
from IPython.display import clear_output


DENOSIDERS = {

    "bm3d" : bm3d.bm3d
}


class ADMM():

    def __init__(self, denoiser="bm3d"):
        
        self.denoiser = DENOSIDERS[denoiser]
        

    def restore(self, y, m, rho=1, tau=0.1, lambd=0.1, mu=0.1, iters=10, sol=None):

        input_size = y.shape

        # INITIALIZE VARIAIBLES
        x = y
        l = np.random.random(input_size)
        s = np.random.random(input_size)

        # INITIALIZE AUXILIAR VARIABLES
        z = np.zeros(input_size)
        u = np.zeros(input_size)
        v = np.zeros(input_size)

        c = 1 / (m + 2*rho)

        for i in range(iters):
            x, l, s, z , u, v = self.step(y, x, l , s, z, u, v, rho, tau, lambd, mu, c)

            if sol is not None:
                clear_output()
                error = np.linalg.norm( sol - x, ord='fro')
                print(f"ITERATION {i} - ERRROR {error}")
                plt.imshow(x, cmap='gray')
                plt.show()

        return x
    
    def step(self, y, x, l , s, z, u , v, rho, tau, lambd, mu, c):

        x = c*(  y + rho*(l + s + z)  - rho*(u + v) )
        l = svd_th(x + u - s, lambd/rho)
        s = soft_th( x + u - l, tau/rho)
        z = self.denoiser( x + v , mu/rho)
        u = u + x - l - s
        v = v + x - z

        return x, l , s , z , u , v





 
