""" the exchange economy from lecture 10

The model class from the lecture notebook, collected in one file so it can be
imported both there and in the problem set. Use it as a template for your own
models. The plotting code for the Edgeworth box is in edgeworth_box.py.

"""

from types import SimpleNamespace
from scipy import optimize
import numpy as np

class ExchangeEconomy:
    """ an exchange economy with two consumers, two goods and Cobb-Douglas preferences """

    def __init__(self,**kwargs):
        """ set the default parameters, then overwrite with any keyword arguments """

        # a. preferences
        self.alpha = 1/3 # the share of her income A spends on good 1
        self.beta = 2/3 # the share of her income B spends on good 1

        # b. endowments (B owns the rest)
        self.w1A = 0.8 # what A owns of good 1
        self.w2A = 0.3 # what A owns of good 2

        # c. settings for the auction algorithm in section 6
        self.nu = 0.5 # step size
        self.tol = 1e-8 # stop when excess demand is this close to zero
        self.maxiter = 500 # give up after this many rounds

        # d. overwrite with keyword arguments, e.g. ExchangeEconomy(alpha=0.50)
        for key,value in kwargs.items():
            setattr(self,key,value) # same as self.key = value

        # e. empty container for the solution
        self.sol = SimpleNamespace()

    def __str__(self):
        """ called when using print """

        text = 'Exchange economy with:\n'
        text += f'  alpha = {self.alpha:.2f} (A spends this share of her income on good 1)\n'
        text += f'  beta  = {self.beta:.2f} (B spends this share of her income on good 1)\n'
        text += f'  wA    = ({self.w1A:.2f},{self.w2A:.2f}) (what A owns)\n'
        text += f'  wB    = ({1-self.w1A:.2f},{1-self.w2A:.2f}) (what B owns)'

        return text

    ##########################
    # preferences and demand #
    ##########################

    def utility_A(self,x1,x2):
        """ A's utility of consuming x1 of good 1 and x2 of good 2 """

        return x1**self.alpha * x2**(1-self.alpha)

    def utility_B(self,x1,x2):
        """ B's utility of consuming x1 of good 1 and x2 of good 2 """

        return x1**self.beta * x2**(1-self.beta)

    def demand_A(self,p1):
        """ A's demand at the price p1, returns (x1,x2) """

        m = p1*self.w1A + self.w2A # the value of what A owns

        return self.alpha*m/p1, (1-self.alpha)*m

    def demand_B(self,p1):
        """ B's demand at the price p1, returns (x1,x2) """

        m = p1*(1-self.w1A) + (1-self.w2A) # the value of what B owns

        return self.beta*m/p1, (1-self.beta)*m

    def excess_demand(self,p1):
        """ how much more of each good is wanted than exists, returns (eps1,eps2) """

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        return x1A+x1B-1.0, x2A+x2B-1.0

    #########################
    # the equilibrium price #
    #########################

    def equilibrium_price(self):
        """ the analytical equilibrium price of good 1 """

        top = self.alpha*self.w2A + self.beta*(1-self.w2A)
        bottom = (1-self.alpha)*self.w1A + (1-self.beta)*(1-self.w1A)

        return top/bottom

    def solve_equilibrium_price(self,bracket=[0.01,100.0]):
        """ the equilibrium price, found numerically as the root of excess demand """

        obj = lambda p1: self.excess_demand(p1)[0] # zero in equilibrium
        result = optimize.root_scalar(obj,bracket=bracket,method='brentq')

        return result.root

    def solve(self,p1=None):
        """ store the price, the allocation and the utilities in self.sol

        Args:

            p1 (float,optional): the price, self.equilibrium_price() is used if None

        Returns:

            (SimpleNamespace): the solution, also stored in self.sol

        """

        if p1 is None: p1 = self.equilibrium_price()

        sol = self.sol
        sol.p1 = p1
        sol.x1A,sol.x2A = self.demand_A(p1)
        sol.x1B,sol.x2B = self.demand_B(p1)
        sol.uA = self.utility_A(sol.x1A,sol.x2A)
        sol.uB = self.utility_B(sol.x1B,sol.x2B)

        return sol

    def tatonnement(self,p1_guess,nu=None,do_print=False):
        """ find the equilibrium price with the auction algorithm in section 6

        Args:

            p1_guess (float): the first price the auctioneer calls
            nu (float,optional): step size, self.nu is used if None
            do_print (bool): print the first rounds

        Returns:

            (ndarray): the price called in each round
            (bool): True if excess demand got below self.tol

        """

        if nu is None: nu = self.nu

        p1 = p1_guess
        path = [p1]

        for k in range(self.maxiter):

            # a. excess demand at the price called
            eps1 = self.excess_demand(p1)[0]

            if do_print and (k < 5 or k%10 == 0):
                print(f'{k:3d}: p1 = {p1:8.5f} -> excess demand = {eps1:9.5f}')

            # b. done?
            if np.abs(eps1) < self.tol:
                if do_print: print(f'{k:3d}: p1 = {p1:8.5f} -> converged')
                return np.array(path),True

            # c. call a new price: up if too much is wanted, down if too little
            p1 = p1 + nu*eps1
            path.append(p1)

            # d. a price can never be negative
            if p1 <= 0: return np.array(path),False

        return np.array(path),False
