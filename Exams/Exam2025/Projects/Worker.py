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

        u_c = np.log(c)
        u_ell = par.nu*ell**(1+par.epsilon)/(1+par.epsilon)
        u = u_c - u_ell
        
        return u
    
    def tax(self,pre_tax_income):

        par = self.par

        tax = par.tau*pre_tax_income + par.zeta

        if not np.isnan(par.kappa):
            tax += par.omega*np.fmax(pre_tax_income-par.kappa,0.0)

        return tax
    
    def income(self,p,ell):

        par = self.par

        return par.w*p*ell

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
        
        if not np.isnan(par.kappa): assert par.zeta <= 0, 'Only lump-sum transfers allowed with tax kink.'

        assert min_ell < par.ell_max, 'No feasible labor supply given parameters.'

        return np.fmax(min_ell,0.0) + 1e-8
    
    def optimal_choice(self,p):

        par = self.par
        opt = SimpleNamespace()

        # a. objective function
        obj = lambda ell: -self.value_of_choice(p,ell)

        # b. bounds and minimization
        bounds = (self.get_min_ell(p),par.ell_max)
        res = minimize_scalar(obj,bounds=bounds,method='bounded')

        # c. results
        opt.ell = res.x
        opt.U = -res.fun
        opt.c = self.post_tax_income(p,opt.ell)

        return opt
    
    def FOC(self,p,ell,omega=0.0):

        par = self.par

        # a. consumption
        c = self.post_tax_income(p,ell)

        # b. marginal utilities
        dU_dc = c**(-1)
        dc_dell = par.w*p*(1-par.tau-omega)
        dU_dell = par.nu*ell**par.epsilon

        # c. FOC
        FOC = dU_dc*dc_dell - dU_dell

        return FOC
    
    def optimal_choice_FOC(self,p):

        par = self.par
        opt = SimpleNamespace()

        # a. no kink
        if np.isnan(par.kappa):

            # i. objective function
            obj = lambda ell: self.FOC(p,ell)

            # ii. bounds and root finding
            min_ell = self.get_min_ell(p)
            max_ell = par.ell_max

            if obj(min_ell)*obj(max_ell) < 0:
                
                bounds = (min_ell,max_ell)
                res = root_scalar(obj,bracket=bounds,method='bisect')
                opt.ell = res.root

            else:

                opt.ell = par.ell_max
            
            opt.U = self.value_of_choice(p,opt.ell)
            opt.c = self.post_tax_income(p,opt.ell)
            opt.section = None

            return opt
        
        else:

            ell_kink = par.kappa/(par.w*p)

            # i. objective functions
            FOC_below = lambda ell: self.FOC(p,ell,omega=0.0)
            FOC_above = lambda ell: self.FOC(p,ell,omega=par.omega)
            
            # ii. bounds and root finding
            min_ell = self.get_min_ell(p)

            # below
            if min_ell < ell_kink and FOC_below(min_ell)*FOC_below(ell_kink) < 0:

                bounds = (min_ell,ell_kink)
                res = root_scalar(FOC_below,bracket=bounds,method='bisect')

                opt.ell_below = res.root
                opt.U_below = self.value_of_choice(p,opt.ell_below)

            else:

                opt.ell_below = np.nan
                opt.U_below = -np.inf

            # above
            if FOC_above(ell_kink)*FOC_above(par.ell_max) < 0:

                bounds = (ell_kink,par.ell_max)
                res = root_scalar(FOC_above,bracket=bounds,method='bisect')

                opt.ell_above = res.root
                opt.U_above = self.value_of_choice(p,opt.ell_above)

            else:

                opt.ell_above = np.nan
                opt.U_above = -np.inf

            # kink
            if min_ell < ell_kink:

                opt.ell_kink = ell_kink
                opt.U_kink = self.value_of_choice(p,opt.ell_kink)

            else:
                
                opt.ell_kink = np.nan
                opt.U_kink = -np.inf

            # iii. choose best
            if opt.U_below >= opt.U_above and opt.U_below >= opt.U_kink:

                opt.ell = opt.ell_below
                opt.U = opt.U_below
                opt.section = 'below'

            elif opt.U_above >= opt.U_below and opt.U_above >= opt.U_kink:
                
                opt.ell = opt.ell_above
                opt.U = opt.U_above
                opt.section = 'above'

            else:

                opt.ell = opt.ell_kink
                opt.U = opt.U_kink
                opt.section = 'kink'
            
            opt.c = self.post_tax_income(p,opt.ell)

        return opt