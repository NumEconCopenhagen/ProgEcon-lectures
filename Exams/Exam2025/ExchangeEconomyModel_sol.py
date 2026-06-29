import numpy as np

from ExchangeEconomyModel import ExchangeEconomyModelClass as ExchangeEconomyModelClass_
  
class ExchangeEconomyModelClass(ExchangeEconomyModelClass_):

    def solve_walras(self,p_guess,print_output=True,method='tatonnement'):

        par = self.par
        sol = self.sol

        # a. initial guess
        p1 = p_guess
        
        # b. iteratre
        k = 0
        self.sol.p1_list = []
        self.sol.eps1_list = []

        while True:

            # i. excess demand
            eps = self.check_market_clearing(p1)

            self.sol.p1_list.append(p1)
            self.sol.eps1_list.append(eps[0])
            
            # ii. check convergence
            if k >= par.K: raise ValueError('max iterations exceeded')

            if np.abs(eps[0]) < par.tol:
                
                sol.p1 = p1
                sol.xA = self.demand_A(p1)
                sol.uA = self.utility_A(sol.xA[0],sol.xA[1])
                sol.uB = self.utility_B(1-sol.xA[0],1-sol.xA[1])

                if print_output:

                    print('\nSolved!')
                    print(f' {k:5d}: p1 = {p1:12.8f}, x1A = {sol.xA[0]:12.8f}, x2A = {sol.xA[1]:12.8f}')
                    print(f' Excess demand of good 1: {eps[0]:14.8f}')
                    print(f' Excess demand of good 2: {eps[1]:14.8f}')

                break

            # iii. print
            if print_output:

                if k < 5 or k%5 == 0:
                    print(f'{k:5d}: p1 = {p1:12.8f} -> excess demand of good 1 -> {eps[0]:14.8f}',end='')
                    print(f', x1A = {self.demand_A(p1)[0]:12.8f}, x2A = {self.demand_A(p1)[1]:12.8f}',end='')
                    print(f', x1B = {self.demand_B(p1)[0]:12.8f}, x2B = {self.demand_B(p1)[1]:12.8f}')
                elif k == 5:
                    print('   ...')

            # iv. p1
            if method == 'tatonnement':
                
                p1 = p1 + par.nu*eps[0]

            elif method == 'newton_raphson':
                
                h = 1e-6

                eps_plus = self.check_market_clearing(p1+h)
                deps1 = (eps_plus[0]-eps[0])/h
    
                p1_new = p1 - par.varphi*eps[0]/deps1

                if p1_new <= 0:
                    p1 = par.iota*p1
                else:
                    p1 = p1_new

            else:
            
                raise ValueError('Unknown method')
            
            # v. increment
            k += 1
