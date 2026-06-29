from types import SimpleNamespace

import numpy as np

from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar

class WorkerClass:

    def __init__(self,par=None):

        # a. setup
        self.setup_worker()

        # b. update parameters
        if not par is None: 
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup_worker(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preferences
        par.nu = 0.015 # weight on labor disutility
        par.epsilon = 1.0 # curvature of labor disutility
        
        # b. productivity and wages
        par.w = 1.0 # wage rate
        par.ps = np.linspace(0.5,3.0,100) # productivities
        par.ell_max = 16.0 # max labor supply
        
        # c. taxes
        par.tau = 0.50 # proportional tax rate
        par.zeta = 0.10 # lump-sum tax
        par.kappa = np.nan # income threshold for top tax
        par.omega = 0.20 # top rate rate
          
    def utility(self,c,ell):

        par = self.par

        pass
        
        return u
    
    def tax(self,pre_tax_income):

        par = self.par

        pass

        return tax
    
    def income(self,p,ell):

        par = self.par

        return None

    def post_tax_income(self,p,ell):

        pre_tax_income = self.income(p,ell)
        tax = self.tax(pre_tax_income)

        return pre_tax_income - tax
    
    def max_post_tax_income(self,p):

        par = self.par
        return self.post_tax_income(p,par.ell_max)

    def value_of_choice(self,p,ell):

        par = self.par

        c = self.post_tax_income(p,ell)
        U = self.utility(c,ell)

        return U
    
    def get_min_ell(self,p):
    
        par = self.par

        min_ell = par.zeta/(par.w*p*(1-par.tau))

        return np.fmax(min_ell,0.0) + 1e-8
    
    def optimal_choice(self,p):

        par = self.par
        opt = SimpleNamespace()

        # a. objective function
        pass

        # b. bounds and minimization
        pass

        # c. results
        opt.ell = res.x
        opt.U = -res.fun
        opt.c = self.post_tax_income(p,opt.ell)

        return opt
    
    def FOC(self,p,ell):

        par = self.par

        pass

        return FOC
    
    def optimal_choice_FOC(self,p):

        par = self.par
        opt = SimpleNamespace()

        pass

        return opt