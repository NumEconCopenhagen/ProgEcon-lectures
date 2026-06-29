from types import SimpleNamespace
from copy import deepcopy
import pickle

import numpy as np
from scipy import optimize

class ProductionEconomyClass():

    def __init__(self,load=None,blank=False,**kwargs):
        """ initialize model """

        if blank:
        
            pass
        
        elif not load is None:
        
            self.load(load)
        
        else:
            
            self.setup()

            # override parameters
            for key,value in kwargs.items():
                self.par.__dict__[key] = value

    def setup(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. demographics
        par.Nw = 99 # number of workers
        par.Nc = 1 # number of capitalists

        # b. parameters
        par.kappa = 0.1 # home production
        par.omega = 10 # disutility of labor supply factor
        par.eta = 1.50 # curvature of disutility of labor supply
        
        # c. production function
        par.alpha = 0.50 # curvature of production function

        # d. grids
        par.num_w = 10 # number of wage grid points
        par.grid_w = np.linspace(0.1,1.5,par.num_w) # wage grid
        par.grid_mkt_clearing = np.zeros(par.num_w) # market clearing results

        # e. solution
        sol.w = 1 # wage
        sol.p = 1 # output price (numeraire)

    ###########
    # workers #
    ###########

    def utility_w(self,c,l):
        """ utility of workers """
        
        par = self.par

        return np.log(c+par.kappa)-par.omega*l**par.eta

    def workers(self):
        """ maximize utility for workers """
        
        sol = self.sol

        p = sol.p
        w = sol.w

        # a. solve
        obj = lambda l: -self.utility_w((w*l)/p,l) # substitute in the budget constraint
        res = optimize.minimize_scalar(obj,bounds=(0,1),method='bounded')
        
        # b. save
        sol.l_w_star = res.x
        sol.c_w_star = (w*sol.l_w_star)/p
        
    ###############
    # capitalists #
    ###############

    def utility_c(self,c,l):
        """ utility of capitalists """

        par = self.par

        return np.log(c+par.kappa)-par.omega*l**par.eta

    def capitalists(self):
        """ maximize utility of capitalists """

        sol = self.sol

        p = sol.p
        w = sol.w
        pi = sol.pi

        # a. solve
        obj = lambda l: -self.utility_c((w*l+pi)/p,l) # subsittute in the budget constraint
        res = optimize.minimize_scalar(obj,bounds=(0,1),method='bounded')
        
        # b. save
        sol.l_c_star = res.x
        sol.c_c_star = (w*sol.l_c_star+pi)/p
        
    ########
    # firm #
    ########

    def firm(self):
        """ maximize firm profits """
        
        par = self.par
        sol = self.sol

        p = sol.p
        w = sol.w

        # a. solve
        f = lambda l: l**par.alpha
        obj = lambda l: -(p*f(l)-w*l)
        x0 = [0.0]
        res = optimize.minimize(obj,x0,bounds=((0,None),),method='L-BFGS-B')
        
        # b. save
        sol.l_star = res.x[0]
        sol.y_star = f(sol.l_star)
        sol.Pi = p*sol.y_star-w*sol.l_star
        
    ####################
    # Walras equibrium #
    ####################

    def evaluate_equilibrium(self):
        """ evaluate equilirium """

        par = self.par
        sol = self.sol

        # a. optimal behavior of firm
        self.firm()
        sol.pi = sol.Pi/par.Nc

        # b. optimal behavior of households
        self.workers()
        self.capitalists()

        # c. market clearing
        sol.goods_mkt_clearing = par.Nw*sol.c_w_star + par.Nc*sol.c_c_star - sol.y_star
        sol.labor_mkt_clearing = par.Nw*sol.l_w_star + par.Nc*sol.l_c_star - sol.l_star
    
    def find_equilibrium(self,do_print=False,do_show_results=True):
        """ find equilibrium wage """

        par = self.par
        sol = self.sol

        # a. grid search
        if do_print: print('grid search:')
        for i,w in enumerate(par.grid_w):
            
            sol.w = w
            self.evaluate_equilibrium()
            par.grid_mkt_clearing[i] = sol.goods_mkt_clearing
            if do_print: print(f' w = {w:.2f} -> {par.grid_mkt_clearing[i]:12.8f}')
        
        if do_print: print('')

        # b. find bounds
        left = np.max(par.grid_w[par.grid_mkt_clearing < 0])
        right = np.min(par.grid_w[par.grid_mkt_clearing > 0])
        if do_print: print(f'equilibrium wage must be in [{left:.2f},{right:.2f}]\n')            

        # c. bisection search
        def obj(w):
            sol.w = w
            self.evaluate_equilibrium()
            return sol.goods_mkt_clearing

        res = optimize.root_scalar(obj,bracket=[left,right],method='bisect')
        sol.w = res.root
        if do_print: print(f'the equilibrium wage is {sol.w:.4f}\n')
        self.evaluate_equilibrium() # final evaluation

        if do_show_results: self.show_results()

    def show_results(self):
        """ show results """

        par = self.par
        sol = self.sol

        u_w = self.utility_w(sol.c_w_star,sol.l_w_star)
        print(f'workers      : c = {sol.c_w_star:6.4f}, l = {sol.l_w_star:6.4f}, u = {u_w:7.4f}')
        u_c = self.utility_c(sol.c_c_star,sol.l_c_star)
        print(f'capitalists  : c = {sol.c_c_star:6.4f}, l = {sol.l_c_star:6.4f}, u = {u_c:7.4f}')        
        print(f'goods market : {sol.goods_mkt_clearing:.8f}')
        print(f'labor market : {sol.labor_mkt_clearing:.8f}')

    #################
    # copy and save #
    #################

    def copy(self):
        """ create a copy of the model """

        new_model = ProductionEconomyClass(blank=True)
        new_model.par = deepcopy(self.par)
        new_model.sol = deepcopy(self.sol)
        return new_model

    def as_dict(self):
        """ convert model to dictionary """

        return {
            'par': self.par.__dict__,
            'sol': self.sol.__dict__
        }
    
    def save(self,filename):
        """ save model """

        savedict = self.as_dict()
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(savedict, f)

    def load(self,filename):
        """ load model """

        with open(f'{filename}.pkl', 'rb') as f:
            loaddict = pickle.load(f)

        self.par = SimpleNamespace(**loaddict['par'])
        self.sol = SimpleNamespace(**loaddict['sol'])
