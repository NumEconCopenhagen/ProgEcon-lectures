import numpy as np
import matplotlib.pyplot as plt

from ASADModel import ASADModelClass as ASADModelClass_

class ASADModelClass(ASADModelClass_):

    def equilibrium(self, pi_e, v):

        p = self.par

        alpha, z = self._alpha_z(v)
        inv_alpha = 1.0 / alpha

        nom = p['pi_star'] - pi_e + inv_alpha * z
        denom = inv_alpha + p['gamma']

        y_star = p['ybar'] + nom / denom
        pi_star = pi_e + p['gamma'] * (y_star - p['ybar'])

        return float(y_star), float(pi_star)

    def simulate(self, rho, eps):

        p = self.par

        # a. setup and allocate
        phi = p['phi']
        T = len(eps)

        v = np.empty(T)
        y_star = np.empty(T)
        pi_star = np.empty(T)
        pi_e = np.empty(T)

        # b. simulate
        pi_e_prev = p['pi_star']
        pi_prev = p['pi_star']
        v_prev = 0.0

        for t in range(T):

            # i. pi_e_t
            if t == 0:
                pi_e_t = p['pi_star']
            else:
                pi_e_t = phi * pi_e_prev + (1-phi) * pi_prev

            pi_e[t] = pi_e_t
            
            # ii. v_t
            v_t = rho * v_prev + eps[t]
            v[t] = v_t

            # iii. (y_t^*, pi_t^*)
            y_star[t], pi_star[t] = self.equilibrium(pi_e_t, v_t)
            
            # iv. update prev
            pi_e_prev = pi_e_t
            pi_prev = pi_star[t]
            v_prev = v_t

        return y_star, pi_star, v

    def moments(self, y, pi):

        p = self.par
        
        sd_y_gap = np.std(y)
        sd_pi = np.std(pi)
        
        corr = np.corrcoef(y, pi)[0,1]

        return sd_y_gap, sd_pi, corr