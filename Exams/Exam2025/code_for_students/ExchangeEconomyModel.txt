from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

class ExchangeEconomyModelClass:

    def __init__(self):
        """ Initialize model """

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preferences
        par.alpha_A = 1.0 # weight on good 1 consumer A
        par.beta_A = (12/37)**3 # weight on good 2 consumer A
        par.rho_A = -2.0 # elasticity parameter A
         
        par.alpha_B = (12/37)**3 # weight on good 1 consumer B
        par.beta_B = 1.0 # weight on good 2 consumer B
        par.rho_B = -2.0 # elasticity parameter B

        # b. endowments
        par.w1A = 1.0-1e-8 # endowment of good 1 consumer A
        par.w2A = 1e-8 # endowment of good 2 consumer A

        # c. solution parameters
        par.tol = 1e-8 # tolerance for convergence
        par.K = 5000 # maximum number of iterations
        
        par.nu = 50.0 # step size for tâtonnement

        par.varphi = 0.1 # dampening factor for Newton-Raphson
        par.iota = 0.99 # price reduction factor if price goes negative

    ######################
    # utility and demand #
    ######################

    def CES_utility(self,x1,x2,alpha,beta,rho):

        return (alpha*x1**rho + beta*x2**rho)**(1/rho)
    
    def CES_indifference(self,u,x1,alpha,beta,rho):

        x2 = np.nan*np.ones_like(x1)

        temp = (u**rho-alpha*x1**rho)/beta
        I = temp >= 0
        x2[I] = temp[I]**(1/rho)

        return x2

    def CES_demand(self,p1,m,alpha,beta,rho):

        sigma = 1/(1-rho)

        fac1 = alpha**sigma*p1**(1-sigma)
        fac2 = beta**sigma
    
        denom = fac1 + fac2
        x1 = fac1/p1*m/denom
        x2 = fac2*m/denom

        assert np.isclose(p1*x1+x2,m), 'budget constraint not satisfied'

        return x1, x2
    
    def utility_A(self,x1A,x2A):

        par = self.par
        return self.CES_utility(x1A,x2A,par.alpha_A,par.beta_A,par.rho_A)

    def x2A_indifference(self,uA,x1A):

        par = self.par
        return self.CES_indifference(uA,x1A,par.alpha_A,par.beta_A,par.rho_A)

    def utility_B(self,x1B,x2B):

        par = self.par
        return self.CES_utility(x1B,x2B,par.alpha_B,par.beta_B,par.rho_B)

    def x2B_indifference(self,uB,x1B):

        par = self.par
        return self.CES_indifference(uB,x1B,par.alpha_B,par.beta_B,par.rho_B)

    def demand_A(self,p1,m=None):

        if m is None: m = p1*self.par.w1A + self.par.w2A
        return self.CES_demand(p1,m,self.par.alpha_A,self.par.beta_A,self.par.rho_A)

    def demand_B(self,p1,m=None):

        if m is None: m = p1*(1-self.par.w1A) + (1-self.par.w2A)
        return self.CES_demand(p1,m,self.par.alpha_B,self.par.beta_B,self.par.rho_B)
    
    ########
    # plot #
    ########

    def create_edgeworthbox(self,figsize=(6,6)):

        par = self.par

        # a. total endowment
        w1bar = 1.0
        w2bar = 1.0

        # b. figure set up
        fig = plt.figure(figsize=figsize, dpi=100)
        ax_A = fig.add_subplot(1,1,1)

        ax_A.set_xlabel('$x_1^A$')
        ax_A.set_ylabel('$x_2^A$')

        temp = ax_A.twinx()
        temp.set_ylabel('$x_2^B$')

        ax_B = temp.twiny()
        ax_B.set_xlabel('$x_1^B$')
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # c. limits
        ax_A.plot([0,w1bar],[0,0],lw=2,color='black') # sides of box
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1]) # figure bigger than box
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        # d. make sure ax_A is on top
        ax_A.set_zorder(ax_B.get_zorder()+1)
        ax_A.patch.set_visible(False)

        return fig, ax_A, ax_B

    def indifference_curve_A(self,ax_A,x1A,x2A,**kwargs):

        uA = self.utility_A(x1A,x2A)
        
        x1A_grid = np.linspace(0.0001,0.9999,1000)
        x2A_grid = self.x2A_indifference(uA,x1A_grid)
        
        I = (x2A_grid > 0) & (x2A_grid < 1) # only inside box
        ax_A.plot(x1A_grid[I],x2A_grid[I],**kwargs)

    def indifference_curve_B(self,ax_B,x1B,x2B,**kwargs):

        uB = self.utility_B(x1B,x2B)
        
        x1B_grid = np.linspace(0.0001,0.9999,1000)
        x2B_grid = self.x2B_indifference(uB,x1B_grid)
        
        I = (x2B_grid > 0) & (x2B_grid < 1) # only inside box
        ax_B.plot(x1B_grid[I],x2B_grid[I],**kwargs)

    def plot_budget_line(self,ax_A):

        par = self.par
        sol = self.sol

        x1A_grid = np.linspace(0,1,100)
        x2_A = par.w2A-sol.p1*(x1A_grid-par.w1A)
        
        I = (x2_A > 0) & (x2_A < 1)
        ax_A.plot(x1A_grid[I],x2_A[I],color='black',ls='--',label='budget line')

    def add_legend(self,ax_A,ax_B,bbox_to_anchor=(0.10,0.60)):

        handles_A, labels_A = ax_A.get_legend_handles_labels()
        handles_B, labels_B = ax_B.get_legend_handles_labels()
        ax_A.legend(handles_A+handles_B,labels_A+labels_B,
                    bbox_to_anchor=bbox_to_anchor,loc='lower left',
                    facecolor='white',framealpha=0.90,fontsize=12)

    ######################
    # Walras equilibrium #
    ######################
    
    def check_market_clearing(self,p1):
        """
        Checks if the market clears for good 1 and 2.
        """

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = (x1A-par.w1A) + x1B-(1-par.w1A)
        eps2 = (x2A-par.w2A) + x2B-(1-par.w2A)

        return eps1,eps2
  
    def solve_walras(self,p_guess,print_output=True,method='tatonnement'):
        
        raise NotImplementedError