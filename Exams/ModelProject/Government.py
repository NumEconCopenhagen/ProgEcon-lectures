from Worker import WorkerClass
import numpy as np
from scipy.optimize import minimize

class GovernmentClass(WorkerClass):

    def __init__(self,par=None):

        # a. defaul setup
        self.setup_worker()
        self.setup_government()

        # b. update parameters
        if not par is None: 
            for k,v in par.items():
                self.par.__dict__[k] = v

        # c. random number generator
        self.rng = np.random.default_rng(12345)

    def setup_government(self):

        par = self.par

        # a. workers
        par.N = 100  # number of workers
        par.sigma_p = 0.3  # std dev of productivity

        # b. pulic good
        par.chi = 50.0 # weight on public good in SWF
        par.eta = 0.1 # curvature of public good in SWF

    def draw_productivities(self):

        par = self.par

        pass

    def solve_workers(self):

        par = self.par
        sol = self.sol

        pass

    def tax_revenue(self):

        par = self.par
        sol = self.sol

        tax_revenue = 0.0

        pass

        return tax_revenue
    
    def SWF(self):

        par = self.par
        sol = self.sol

        G =  self.tax_revenue()
        if G < 0:
            SWF = np.nan
        else:
            SWF = None

        return SWF
    
    def optimal_taxes(self,tau,zeta):

        par = self.par

        # a. objective function
        def obj(x):

            par.tau = x[0]
            par.zeta = x[1]

            pass

            return -SWF
        
        # b. optimization               
        pass

        # c. results
        pass

        return None