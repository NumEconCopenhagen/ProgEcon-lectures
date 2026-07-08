import numpy as np
import matplotlib.pyplot as plt

class ASADModelClass:

    def __init__(self, par=None):
    
        par = dict(
            ybar    = 1.0,
            pi_star = 0.02,
            b       = 0.6,
            a1      = 1.5,
            a2      = 0.10,
            gamma   = 4.0,
            phi     = 0.6
        )

        self.par = par.copy()

    def _alpha_z(self, v):

        p = self.par

        alpha = p['b'] * p['a1'] / (1.0 + p['b'] * p['a2'])
        z = v / (1.0 + p['b'] * p['a2'])

        return alpha, z

    def AD_curve(self, y, v):

        p = self.par
        alpha, z = self._alpha_z(v)
        inv_alpha = 1.0 / alpha
        return p['pi_star'] - inv_alpha * ((y - p['ybar']) - z)

    def SRAS_curve(self, y, pi_e):
        
        p = self.par
        return pi_e + p['gamma'] * (y - p['ybar'])

    # analytical equilibrium y_t^*, pi_t^* given pi_e and v
    def equilibrium(self, pi_e, v):
        raise NotImplementedError

    # simulation
    def simulate(self, rho, eps):
        raise NotImplementedError

    # compute sd(y_gap), sd(pi), corr(y_gap, pi)
    def moments(self, y, pi):
        raise NotImplementedError
