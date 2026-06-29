from types import SimpleNamespace

import numpy as np

class LaborMarketModelClass:
    """Labor market simulator with two worker types (low/high) used in Problem 3.

    Type coding:
        0 = low-skilled
        1 = high-skilled
    """

    def __init__(self, par=None):

        p = SimpleNamespace()

        # a. population
        p.N = 50_000
        p.share_high = 0.40

        # b. labor market
        p.sep_rate = 0.02  # separation probability per period

        # job finding (type-specific)
        p.f0_L = 0.25
        p.f0_H = 0.45
        p.lambda_L = 0.08
        p.lambda_H = 0.03

        # c. human capital (log)
        p.rho = 0.97
        p.mu_L = 0.00
        p.mu_H = 0.50
        p.sigma_h = 0.06
        p.delta_u = 0.03  # depreciation during unemployment

        # d. income
        p.a_L = 2.40
        p.a_H = 2.80
        p.sigma_y = 0.20
        p.benefit_replace = 0.60  # replacement rate b

        # e. simulation controls
        p.seed = 123

        # f. override defaults with user-specified parameters
        if par is not None:
            for k, v in par.items():
                setattr(p, k, v)

        self.par = p

    def _type_arrays(self, types):
        """Return type-specific arrays (mu, a, f0, lambda) for each individual."""

        p = self.par
        mu = np.where(types == 1, p.mu_H, p.mu_L)
        a = np.where(types == 1, p.a_H, p.a_L)
        f0 = np.where(types == 1, p.f0_H, p.f0_L)
        lam = np.where(types == 1, p.lambda_H, p.lambda_L)

        return mu, a, f0, lam

    def job_finding_prob(self, types, duration):
        """Job-finding probability for unemployed workers.

        Implement:
            f_s(d) = min{1, f0_s * exp(-lambda_s*(d-1))}  for d>=1,
        and then clip from below at 0.05.

        You can assume this is only called for unemployed individuals (duration>=1),
        but write robust code (e.g. handle duration<=0 safely).
        """

        # TODO: implement
        raise NotImplementedError

    def setup_states(self, rng):
        """Initialize (types, employed, duration, k).

        Initialization (as in the problem statement):
        - types: high-skilled with probability share_high
        - employed: employed with probability 0.92
        - duration: if unemployed draw a duration in {1,...,6}, else 0
        - k: draw from a stationary distribution of the AR(1) around mu_s
        """

        p = self.par
        N = p.N

        # a. types
        types = (rng.random(N) < p.share_high).astype(np.int64)

        # b. employment status and duration
        employed = rng.random(N) < 0.92
        duration = np.zeros(N, dtype=np.int64)
        duration[~employed] = rng.integers(1, 7, size=(~employed).sum(), dtype=np.int64)

        # c. human capital k
        mu, _, _, _ = self._type_arrays(types)
        sigma_stationary = p.sigma_h / np.sqrt(max(1e-12, 1 - p.rho**2))
        k = mu + sigma_stationary * rng.standard_normal(N)

        return types, employed, duration, k

    def step(self, types, employed, duration, k, rng):
        """One period update (vectorized).

        Returns:
            y : income in the CURRENT period (based on current states)
            employed_next, duration_next, k_next : next-period states
        """

        # TODO: implement
        raise NotImplementedError

    def income_moments(self, y):
        """A small set of moments for log income."""

        y = np.asarray(y)
        y = y[np.isfinite(y)]
        logy = np.log(np.maximum(y, 1e-12))
        return {
            'mean_logy': float(logy.mean()),
            'std_logy': float(logy.std()),
        }

    def simulate(self, T=3_000, store_last=200):
        """Simulate forward for T periods.

        The function must store a small panel with the LAST `store_last` periods,
        so you can compare the distribution in periods -200:-100 vs -100:.

        Return a dictionary with (at least):
            - types: (N,) array with worker types (0/1)
            - y_last: (store_last, N) income for the last periods
            - employed_last: (store_last, N) employment status for the last periods
            - duration_last: (store_last, N) unemployment duration for the last periods
            - u_rate, mean_logy, std_logy: time series of selected moments (length T)
        """

        # TODO: implement
        raise NotImplementedError
