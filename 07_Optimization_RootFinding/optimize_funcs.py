# optimize_funcs.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- module-level state captured by the callback ---
x1s = []
x2s = []
fs  = []
nit = 0
init_guess = None
obj = None

def init_collector(obj_func, init_guess_in):
    """Initialize the iteration history before calling scipy.optimize.minimize."""
    global x1s, x2s, fs, nit, obj, init_guess
    obj = obj_func
    init_guess = np.asarray(init_guess_in, dtype=float)

    x1s = [init_guess[0]]
    x2s = [init_guess[1]]
    fs  = [obj(init_guess)]  # objective at initial guess
    nit = 0

def collect(x):
    """SLSQP callback: collects iterate x and objective value."""
    global x1s, x2s, fs, nit, obj
    x1s.append(x[0])
    x2s.append(x[1])
    fs.append(obj(x))
    nit += 1

def convergence_steps(plot_utility=False, ax=None, show=True, return_data=False):
    """
    Plot objective (or utility) by iteration using collected history.
    Set return_data=True to get (fig, ax, y); otherwise returns None.
    """
    if not fs:
        raise RuntimeError(
            "No iterations collected. Call init_collector(...) and run "
            "minimize(..., callback=collect) first."
        )

    y = -np.asarray(fs) if plot_utility else np.asarray(fs)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(np.arange(len(y)), y, '-o', ms=4)
    ax.set_xlabel('iteration')
    ax.set_ylabel('utility u' if plot_utility else 'objective')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    if show:
        plt.show()

    if return_data:
        return fig, ax, y
    # else: return None implicitly
