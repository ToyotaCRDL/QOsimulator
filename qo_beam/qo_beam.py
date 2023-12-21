import os, sys
import copy
import numpy as np
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt

class QObeam():
    '''
    Generate Gaussian photon beam
    '''
    def __init__(self, N, L, x0, y0, kx0, ky0,sigma_x, sigma_y, n_bullets=1, theta_pol=0):
        self.name = "1ph"
        self.N = N
        self.L = L
        self.x0 = x0
        self.y0 = y0
        self.kx0 = kx0
        self.ky0 = ky0
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.n_bullets = n_bullets
        self.theta_pol = theta_pol

    def gen_photon_ck(self, kx, ky):
        ck_t0 = np.zeros(self.N,dtype = 'complex_')
        x0 = self.x0
        y0 = self.y0
        sigma_x = self.sigma_x
        sigma_y = self.sigma_y
        kx0 = self.kx0
        ky0 = self.ky0
        interval_x = 4.0*np.sign(kx0)*sigma_x
        interval_y = 4.0*np.sign(ky0)*sigma_y
        x0s = [x0-nn*interval_x for nn in range(self.n_bullets)]
        y0s = [y0-nn*interval_y for nn in range(self.n_bullets)]
        for nn in range(self.n_bullets):
            ck_t0 +=  self._gen_photon_ck(kx, ky, x0s[nn], y0s[nn])
        ### Normalize
        norm = np.sum(np.abs(ck_t0)**2.0)
        ck_t0 = ck_t0/sqrt(norm)
        return ck_t0

    def _gen_photon_ck(self, kx, ky, x0, y0):
        N = self.N
        L = self.L
        kx0 = self.kx0
        ky0 = self.ky0
        sigma_x = self.sigma_x
        sigma_y = self.sigma_y
        ##
        denom = (2*pi*sigma_x/L[0])**0.5 * (pi)**(-0.25) * (2*pi*sigma_y/L[1])**0.5 * (pi)**(-0.25)
        ck_t0 = np.zeros(N,dtype = 'complex_')
        for i in range(N[0]):
            for j in range(N[1]):
                fac = denom * exp(-1.0j*((kx[i]*x0 + ky[j]*y0)))
                ck_t0[i, j] = fac * exp(-0.5*sigma_x*sigma_x*(kx[i] - kx0)**2.0 - 0.5*sigma_y*sigma_y*(ky[j] - ky0)**2.0)
        return ck_t0
