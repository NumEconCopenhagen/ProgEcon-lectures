from types import SimpleNamespace
import numpy as np

from LaborMarketModel import LaborMarketModelClass as LaborMarketModelClass_

class LaborMarketModelClass(LaborMarketModelClass_):

    ###########
    # helpers #
    ###########

    def job_finding_prob(self, types, duration):
        """Job-finding probability.

        f_s(d) = min{1, f0_s * exp(-lambda_s*(d-1))} for d>=1,
        then clipped from below at 0.05.
        """

        _, _, f0, lam = self._type_arrays(types)

        d = np.maximum(np.asarray(duration), 1)
        prob = f0 * np.exp(-lam * (d - 1.0))

        return np.clip(prob, 0.05, 1.0)

    ##############
    # simulation #
    ##############

    def step(self, types, employed, duration, k, rng):
        """One period update (vectorized)."""

        p = self.par
        N = p.N

        mu, a, _, _ = self._type_arrays(types)

        # a. draw shocks efficiently (two calls per period)
        shocks = rng.standard_normal((2, N))
        xi = shocks[0]
        eta = shocks[1]

        uniforms = rng.random((2, N))
        u_sep = uniforms[0]
        u_find = uniforms[1]

        # b. income in current period
        wage = np.exp(a + k + p.sigma_y * xi)
        benefit = p.benefit_replace * np.exp(a + k)
        y = np.where(employed, wage, benefit)

        # c.human capital update (depends on CURRENT employment)
        k_next = mu + p.rho * (k - mu) + p.sigma_h * eta - p.delta_u * (~employed)
        k_next = np.clip(k_next, -4.0, 6.0)

        # d. transitions
        employed_next = employed.copy()
        duration_next = np.zeros_like(duration)

        # i) separation from employment
        sep = employed & (u_sep < p.sep_rate)
        employed_next[sep] = False
        duration_next[sep] = 1

        # ii) job finding from unemployment
        unemployed = ~employed

        if unemployed.any():

            hazard = self.job_finding_prob(
                types=types[unemployed],
                duration=duration[unemployed],
            )
            found = u_find[unemployed] < hazard

            employed_next[unemployed] = found

            stay_u = unemployed.copy()
            stay_u[unemployed] = ~found
            duration_next[stay_u] = duration[stay_u] + 1

        return y, employed_next, duration_next, k_next

    def simulate(self, T=3_000, store_last=200, store_k=True):
        """Simulate forward and store the last `store_last` periods.

        Returns a dictionary with:
            - types: (N,)
            - y_last: (store_last, N)
            - employed_last: (store_last, N)
            - duration_last: (store_last, N)
            - k_last: (store_last, N) if store_k=True
            - u_rate, mean_logy, std_logy: time series of selected moments (length T)
        """

        p = self.par
        rng = np.random.default_rng(p.seed)

        # a. setup
        types, employed, duration, k = self.setup_states(rng)

        T = int(T)
        store_last = int(store_last)

        u_rate = np.empty(T, dtype=np.float64)
        mean_logy = np.empty(T, dtype=np.float64)
        std_logy = np.empty(T, dtype=np.float64)

        # b. panel storage for last periods (time x individuals)
        y_last = np.empty((store_last, p.N), dtype=np.float64)
        employed_last = np.empty((store_last, p.N), dtype=np.int64)
        duration_last = np.empty((store_last, p.N), dtype=np.int64)
        k_last = np.empty((store_last, p.N), dtype=np.float64) if store_k else None

        t0 = max(0, T - store_last)  # first period to store (0-indexed)

        # c. simulate forward
        for t in range(T):

            # i. current-period outcomes + next-period states
            y, employed_next, duration_next, k_next = self.step(
                types=types,
                employed=employed,
                duration=duration,
                k=k,
                rng=rng,
            )

            # ii. selected moments (CURRENT period)
            u_rate[t] = (~employed).mean()
            mom = self.income_moments(y)
            mean_logy[t] = mom['mean_logy']
            std_logy[t] = mom['std_logy']

            # iii. store panel data for the last periods (CURRENT period)
            if t >= t0:
                j = t - t0
                y_last[j, :] = y
                employed_last[j, :] = employed.astype(np.int64)
                duration_last[j, :] = duration
                if store_k:
                    k_last[j, :] = k

            # iv. advance states
            employed, duration, k = employed_next, duration_next, k_next

        out = {
            'T': int(T),
            'store_last': int(store_last),
            'types': types,
            'y_last': y_last,
            'employed_last': employed_last,
            'duration_last': duration_last,
            'u_rate': u_rate,
            'mean_logy': mean_logy,
            'std_logy': std_logy,
        }

        if store_k: out['k_last'] = k_last

        return out
