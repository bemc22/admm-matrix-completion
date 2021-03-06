import numpy as np
import bm3d
import matplotlib.pyplot as plt

from core.utils import svd_th, svd_est, soft_th, psnr
import time
from IPython.display import clear_output
from skimage.metrics import structural_similarity as ssim

DENOSIDERS = {
    "bm3d" : bm3d.bm3d
}


class ADMM():

    def __init__(self, denoiser="bm3d"):
        
        self.denoiser = DENOSIDERS[denoiser]
        

    def restore(self, y, m, rho=0.2, tau=0.01, lambd=1e-4, mu='auto', iters=10, sol=None):

        input_size = y.shape

        if mu == 'auto':
            mu = np.std(-m)*rho

        # INITIALIZE VARIAIBLES
        x = self.denoiser(y, mu)*(1-m) + y
        l = np.random.random(input_size)
        s = np.random.random(input_size)

        # INITIALIZE AUXILIAR VARIABLES
        z = np.zeros(input_size)
        u = np.zeros(input_size)    
        v = np.zeros(input_size)

        c = 1 / (m + 2*rho)

        for i in range(iters):
            t = time.time()
            x, l, s, z , u, v = self.step(y, x, l , s, z, u, v, rho, tau, lambd, mu, c)
            
            if sol is not None:
                clear_output()
                error = round(np.linalg.norm( sol - x, ord='fro'),2)
                value_psnr = round(psnr(sol, x),2)
                value_ssim = round(ssim(sol, x, data_range=1),2)
                print(f"iteration {i} | error {error} | psnr {value_psnr} | ssim {value_ssim} | time {np.round(time.time() - t, 4)}")
                
                fig , axs = plt.subplots(1, 2)
                
                axs[0].imshow(x, cmap='gray')
                axs[1].imshow(np.abs(sol - x))
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





 
