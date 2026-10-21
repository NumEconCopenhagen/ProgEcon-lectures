"""The basic Solow model (Lecture 11).

The model is a class, as in Lecture 3: the parameters are attributes and the
equations are methods. The notebook Solow_model.ipynb imports it with

    from solow import SolowModel

Use this file as a template for your own models, e.g. in the model project.
"""

import numpy as np
from scipy import optimize


class SolowModel:
    """The basic Solow model with Cobb-Douglas production, y = k**alpha."""

    def __init__(self, alpha=1/3, s=0.20, delta=0.30, k0=0.10, T=50):
        """Store the parameters and the simulation settings."""

        self.alpha = alpha   # capital share in production
        self.s = s           # savings rate
        self.delta = delta   # depreciation rate
        self.k0 = k0         # initial capital per worker
        self.T = T           # number of periods to simulate

    def f(self, k):
        """Output per worker, y = k**alpha."""
        return k**self.alpha

    def k_next(self, k):
        """Capital per worker next period, s*f(k) + (1-delta)*k."""
        return self.s*self.f(k) + (1 - self.delta)*k

    def steady_state(self):
        """Steady state from the formula; returns (k, y, c)."""
        k = (self.s/self.delta)**(1/(1 - self.alpha))
        y = self.f(k)
        c = (1 - self.s)*y
        return k, y, c

    def steady_state_gap(self, k):
        """k_next(k) - k: zero in the steady state."""
        return self.k_next(k) - k

    def solve_steady_state(self):
        """Steady-state capital found with a root-finder (Lecture 7)."""
        result = optimize.root_scalar(self.steady_state_gap, bracket=[1e-6, 100.0], method='brentq')
        if not result.converged:
            raise RuntimeError(result.flag)
        return result.root

    def simulate(self, s_path=None):
        """Simulate T periods from k0.

        s_path is an optional array with one savings rate per period; without it,
        the savings rate self.s is used in every period. The results are stored in
        the arrays self.k_path, self.y_path, self.i_path and self.c_path.
        """

        if s_path is None:
            s_path = np.full(self.T, self.s)

        k = np.empty(self.T)
        y = np.empty(self.T)
        i = np.empty(self.T)
        c = np.empty(self.T)

        k[0] = self.k0
        for t in range(self.T):
            y[t] = self.f(k[t])
            i[t] = s_path[t]*y[t]
            c[t] = y[t] - i[t]
            if t < self.T - 1:
                k[t+1] = i[t] + (1 - self.delta)*k[t]

        self.k_path = k
        self.y_path = y
        self.i_path = i
        self.c_path = c
