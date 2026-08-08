""" the Solow model from lecture 9

The model class from the lecture notebook, collected in one file so it can be
imported both there and in the problem set. Use it as a template for your own models.

"""

from types import SimpleNamespace
from scipy import optimize
import numpy as np

class SolowModel:
    """ the Solow model with a Cobb-Douglas production function """

    def __init__(self,**kwargs):
        """ set the default parameters, then overwrite with any keyword arguments """

        # a. parameters
        self.alpha = 1/3 # capital share in production
        self.s = 0.20 # savings rate
        self.delta = 0.30 # depreciation rate

        # b. simulation settings
        self.k0 = 0.10 # initial capital per worker
        self.T = 50 # number of periods

        # c. overwrite with keyword arguments, e.g. SolowModel(s=0.30)
        for key,value in kwargs.items():
            setattr(self,key,value) # same as self.key = value

        # d. empty container for simulation results
        self.sim = SimpleNamespace()

    def __str__(self):
        """ called when using print """

        text = 'Solow model with:\n'
        text += f'  alpha = {self.alpha:.2f} (capital share)\n'
        text += f'  s     = {self.s:.2f} (savings rate)\n'
        text += f'  delta = {self.delta:.2f} (depreciation rate)\n'
        text += f'  k0    = {self.k0:.2f} (initial capital per worker)\n'
        text += f'  T     = {self.T} (number of periods)'

        return text

    def f(self,k):
        """ output per worker, y = f(k) = k**alpha """

        return k**self.alpha

    def k_next(self,k,s=None):
        """ capital per worker in the next period, k_next = s*f(k) + (1-delta)*k

        Args:

            k (float or ndarray): capital per worker today
            s (float,optional): savings rate, self.s is used if None

        Returns:

            (float or ndarray): capital per worker next period

        """

        if s is None: s = self.s

        return s*self.f(k) + (1-self.delta)*k

    def steady_state(self):
        """ analytical steady state, returns (k,y,c) """

        k = (self.s/self.delta)**(1/(1-self.alpha))
        y = self.f(k)
        c = (1-self.s)*y

        return k,y,c

    def solve_steady_state(self):
        """ numerical steady state, the k where k_next(k)-k = 0 """

        obj = lambda k: self.k_next(k)-k # zero in the steady state
        result = optimize.root_scalar(obj,bracket=[1e-8,1e6],method='brentq')

        return result.root

    def simulate(self,s=None,k0=None,T=None):
        """ simulate the model forward in time

        Args:

            s (float or ndarray,optional): savings rate, a number (constant) or
                an array with T elements (time-varying), self.s is used if None
            k0 (float,optional): initial capital per worker, self.k0 is used if None
            T (int,optional): number of periods, self.T is used if None

        Returns:

            (SimpleNamespace): the simulated paths, also stored in self.sim

        """

        # a. settings
        if s is None: s = self.s
        if k0 is None: k0 = self.k0
        if T is None: T = self.T

        # b. savings rate in each period (np.ndim(s) == 0 means s is a single number)
        s_vec = np.full(T,s) if np.ndim(s) == 0 else np.asarray(s,dtype=float)
        assert s_vec.size == T, f'the savings rate must have {T} elements, but has {s_vec.size}'

        # c. allocate memory
        k = np.empty(T) # capital per worker
        y = np.empty(T) # output per worker
        i = np.empty(T) # investment per worker
        c = np.empty(T) # consumption per worker

        # d. loop forward in time
        k[0] = k0
        for t in range(T):

            y[t] = self.f(k[t]) # production
            i[t] = s_vec[t]*y[t] # investment
            c[t] = y[t]-i[t] # consumption

            if t < T-1: # the law of motion, same as self.k_next(k[t],s_vec[t])
                k[t+1] = i[t] + (1-self.delta)*k[t]

        # e. store the results
        self.sim.k = k
        self.sim.y = y
        self.sim.i = i
        self.sim.c = c
        self.sim.s = s_vec

        return self.sim
