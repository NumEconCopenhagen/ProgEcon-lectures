import numpy as np
from scipy import optimize

from LaborSupplyModel import LaborSupplyModelClass as LaborSupplyModelClass_

class LaborSupplyModelClass(LaborSupplyModelClass_):

    ########################
    # preferences and budget
    ########################

    def earnings(self, l, WageSchedule='baseline'):
        """Earnings (excluding non-labor income y0)."""

        p = self.par
        l = np.asarray(l)

        if WageSchedule == 'baseline':

            e = p.w * l

        elif WageSchedule == 'overtime':
            
            # earnings = w*l for l<=l_bar;
            #          = w*l_bar + w*(1+delta)*(l-l_bar) for l>l_bar
            e = p.w * l + p.w * p.delta * np.maximum(0.0, l - p.l_bar)

        else:

            raise ValueError("WageSchedule must be one of 'baseline' or 'overtime' ")

        return e

    def consumption(self, l, WageSchedule='baseline'):

        p = self.par
        c = p.y0 + self.earnings(l, WageSchedule=WageSchedule)
        return c

    def utility(self, l, WageSchedule='baseline'):
        
        p = self.par
        l = np.asarray(l)
        c = self.consumption(l, WageSchedule=WageSchedule)

        # guard against invalid consumption
        u = np.full_like(c, fill_value=-np.inf, dtype=float)
        valid = (c > 0.0) & (l >= 0.0) & (l <= p.l_max)
        if not np.any(valid):
            return float(u) if np.isscalar(l) else u

        c_v = c[valid]
        if np.isclose(p.sigma, 1.0):
            u_c = np.log(c_v)
        else:
            u_c = (c_v ** (1.0 - p.sigma) - 1.0) / (1.0 - p.sigma)

        u_l = -p.psi * (l[valid] ** (1.0 + p.eta)) / (1.0 + p.eta)
        u[valid] = u_c + u_l

        return float(u) if np.isscalar(l) else u

    def dc_dl(self, l, WageSchedule='baseline'):
        """Marginal consumption with respect to hours (piecewise constant)."""

        p = self.par
        l = np.asarray(l)

        if WageSchedule == 'baseline':

            d = p.w * np.ones_like(l, dtype=float)

        elif WageSchedule == 'overtime':

            d = np.where(l <= p.l_bar, p.w, p.w * (1.0 + p.delta)).astype(float)

        else:

            raise ValueError("WageSchedule must be one of 'baseline' or 'overtime'")

        return float(d) if np.isscalar(l) else d

    def foc(self, l, WageSchedule='baseline'):

        p = self.par
        l = np.asarray(l)

        c = self.consumption(l, WageSchedule=WageSchedule)
        dc = self.dc_dl(l, WageSchedule=WageSchedule)

        if np.isclose(p.sigma, 1.0):
            u_c = 1.0 / c
        else:
            u_c = c ** (-p.sigma)

        foc = u_c * dc - p.psi * (l ** p.eta)

        return float(foc) if np.isscalar(l) else foc

    ######################
    # numerical solution #
    ######################

    def solve_grid(self, WageSchedule='baseline', n=2000):

        p = self.par

        grid = np.linspace(0.0, p.l_max, int(n))
        u = self.utility(grid, WageSchedule=WageSchedule)
        
        i = int(np.nanargmax(u))
        l_star = float(grid[i])

        return {
            'l_star': l_star,
            'c_star': float(self.consumption(l_star, WageSchedule=WageSchedule)),
            'u_star': float(self.utility(l_star, WageSchedule=WageSchedule)),
            'grid': grid,
            'u_grid': u,
        }

    def solve_local_opt(self, l0, WageSchedule='baseline'):
        
        p = self.par
        l0 = float(np.clip(l0, 0.0, p.l_max))

        def obj(x):
            return -float(self.utility(float(x[0]), WageSchedule=WageSchedule))

        res = optimize.minimize(
            obj,
            x0=np.array([l0]),
            method='L-BFGS-B',
            bounds=[(0.0, p.l_max)],
        )

        l_star = float(res.x[0])
        return {
            'l_star': l_star,
            'c_star': float(self.consumption(l_star, WageSchedule=WageSchedule)),
            'u_star': float(self.utility(l_star, WageSchedule=WageSchedule)),
            'success': bool(res.success),
            'message': res.message,
        }
    
    def solve_local_FOC(self, WageSchedule='baseline'):
        
        p = self.par

        def foc_baseline(l):
            return self.foc(l, WageSchedule=WageSchedule)

        sol_root = optimize.root_scalar(foc_baseline, bracket=(1e-8, p.l_max-1e-8), method='brentq')

        if not sol_root.converged:
            
            return {
                'l_star': np.nan,
                'c_star': np.nan,
                'u_star': np.nan,
                'success': False,
                'message': sol_root.flag,
            }
        
        l_star = float(sol_root.root)
        
        return {
            'l_star': l_star,
            'c_star': float(self.consumption(l_star, WageSchedule=WageSchedule)),
            'u_star': float(self.utility(l_star, WageSchedule=WageSchedule)),
            'success': True,
            'message': sol_root.flag,
        }
        

    def solve_global(self, WageSchedule='baseline'):
        """Global maximizer by splitting at kinks and optimizing within each region."""

        p = self.par

        if WageSchedule == 'baseline':
            knots = []
        elif WageSchedule == 'overtime':
            knots = [p.l_bar]
        else:
            raise ValueError("WageSchedule must be one of 'baseline' or 'overtime'")

        # candidate intervals
        points = [0.0] + knots + [p.l_max]
        intervals = [(points[i], points[i+1]) for i in range(len(points)-1)]

        candidates = set(points)

        def obj_scalar(x):
            return -float(self.utility(float(x), WageSchedule=WageSchedule))

        for a, b in intervals:
            
            if b - a <= 1e-12: continue
            
            # within each region the function is typically well-behaved (often concave)
            res = optimize.minimize_scalar(obj_scalar, bounds=(a, b), method='bounded')
            candidates.add(float(res.x))

        # pick best
        cand_list = sorted(candidates)
        u_vals = np.array([self.utility(x, WageSchedule=WageSchedule) for x in cand_list], dtype=float)
        i = int(np.nanargmax(u_vals))
        l_star = float(cand_list[i])

        return {
            'l_star': l_star,
            'c_star': float(self.consumption(l_star, WageSchedule=WageSchedule)),
            'u_star': float(self.utility(l_star, WageSchedule=WageSchedule)),
            'candidates': cand_list,
            'u_candidates': u_vals,
        }
