"""The exchange economy (Lecture 12).

Two consumers, A and B, two goods, Cobb-Douglas preferences. There is one unit
of each good in total, and good 2 is the numeraire (p2 = 1). The notebook
ExchangeEconomy.ipynb imports the class with

    from exchange import ExchangeEconomy

The plotting code for the Edgeworth box is in edgeworth_box.py.
"""

from scipy import optimize


class ExchangeEconomy:
    """An exchange economy with two consumers and two goods."""

    def __init__(self, alpha=1/3, beta=2/3, w1A=0.8, w2A=0.3):
        """Store the parameters; B owns what A does not."""

        self.alpha = alpha   # share of her income A spends on good 1
        self.beta = beta     # share of her income B spends on good 1
        self.w1A = w1A       # what A owns of good 1
        self.w2A = w2A       # what A owns of good 2

    def utility_A(self, x1, x2):
        return x1**self.alpha * x2**(1 - self.alpha)

    def utility_B(self, x1, x2):
        return x1**self.beta * x2**(1 - self.beta)

    def demand_A(self, p1):
        """A's demand at the price p1; returns (x1, x2)."""
        m = p1*self.w1A + self.w2A                  # the value of what A owns
        return self.alpha*m/p1, (1 - self.alpha)*m

    def demand_B(self, p1):
        """B's demand at the price p1; returns (x1, x2)."""
        m = p1*(1 - self.w1A) + (1 - self.w2A)      # the value of what B owns
        return self.beta*m/p1, (1 - self.beta)*m

    def excess_demand(self, p1):
        """Wanted minus available for both goods; returns (eps1, eps2)."""
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)
        return x1A + x1B - 1, x2A + x2B - 1

    def excess_demand_good1(self, p1):
        """Excess demand for good 1 only: zero at the equilibrium price."""
        return self.excess_demand(p1)[0]

    def equilibrium_price(self):
        """The equilibrium price from the formula."""
        top = self.alpha*self.w2A + self.beta*(1 - self.w2A)
        bottom = (1 - self.alpha)*self.w1A + (1 - self.beta)*(1 - self.w1A)
        return top/bottom

    def solve_equilibrium_price(self):
        """The equilibrium price found with a root-finder (Lecture 7)."""
        result = optimize.root_scalar(self.excess_demand_good1, bracket=[0.01, 100.0], method='brentq')
        if not result.converged:
            raise RuntimeError(result.flag)
        return result.root
