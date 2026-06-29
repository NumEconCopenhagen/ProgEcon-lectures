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

        par.ps = np.exp(self.rng.normal(-0.5*par.sigma_p**2,par.sigma_p,par.N))

    def solve_workers(self):

        par = self.par
        sol = self.sol

        sol.ell_opt = np.zeros(par.N)
        sol.U_opt = np.zeros(par.N)
        sol.c_opt = np.zeros(par.N)
        sol.above_kink = np.zeros(par.N,dtype=bool)
        sol.at_kink = np.zeros(par.N,dtype=bool)
        sol.below_kink = np.zeros(par.N,dtype=bool)

        for i in range(par.N):

            p = par.ps[i]
            sol_i = self.optimal_choice_FOC(p)

            sol.ell_opt[i] = sol_i.ell
            sol.U_opt[i] = sol_i.U
            sol.c_opt[i] = sol_i.c
            sol.above_kink[i] = (sol_i.section == 'above')
            sol.at_kink[i] = (sol_i.section == 'kink')
            sol.below_kink[i] = (sol_i.section == 'below')

    def tax_revenue(self):

        par = self.par
        sol = self.sol

        tax_revenue = 0.0

        for i in range(par.N):

            p = par.ps[i]
            ell = sol.ell_opt[i]

            pre_tax_income = self.income(p,ell)
            tax = self.tax(pre_tax_income)

            tax_revenue += tax

        return tax_revenue
    
    def SWF(self):

        par = self.par
        sol = self.sol

        G =  self.tax_revenue()
        if G < 0:
            SWF = np.nan
        else:
            SWF = par.chi*G**par.eta + np.sum(sol.U_opt)

        return SWF
    
    def optimal_taxes(self,tau,zeta,min_zeta=None,max_zeta=None):

        par = self.par

        # a. objective function
        def obj(x):

            par.tau = x[0]
            par.zeta = x[1]

            # i. constraint on zeta
            p_min = par.ps.min()
            max_zeta = self.max_post_tax_income(p_min)
            if par.zeta > max_zeta: return -np.inf
            
            # ii. solve workers
            self.solve_workers()

            # iii. check G
            G = self.tax_revenue()
            if G < 0: return -np.inf

            # iv. social welfare
            SWF = self.SWF()

            return -SWF
        
        # b. optimization               
        res = minimize(obj,[tau,zeta],method='Nelder-Mead',bounds=[(0.01,0.99),(-1.0,1.0)])
        obj(res.x)

        # c. results
        print(f'Optimal tau: {res.x[0]:.4f}, Optimal zeta: {res.x[1]:.4f}, SWF: {-res.fun:.4f}')