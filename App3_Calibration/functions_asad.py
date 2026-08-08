import numpy as np
import matplotlib.pyplot as plt


# Baseline parameters (same as in Lecture 11)
par = dict(
    # steady state
    ybar        = 1.0,
    pi_star     = 0.005, # quarterly rate
    # IS + Taylor (used in flexible-rate AD)
    b   = 0.6,
    a1  = 1.7,    # response to inflation gap (>1)
    a2  = 0.10,   # response to output gap

    # Phillips (SRAS)
    gamma = 0.4,

    # Shock dynamics (for dynamics section)
    delta   = 0.90,   # demand AR(1) persistence
    omega   = 0.15,   # supply AR(1) persistence
    sigma_x = 0.01,   # demand innovation std
    sigma_c = 0.005   # supply innovation std
)



def ad_curve(y, p, v):
    alpha_val = p["b"]*(p["a1"]-1.0)/(1.0 + p["b"]*p["a2"])
    z_t = v/(1.0 + p["b"]*p["a2"])
    return p["pi_star"] - ((y - p["ybar"]) - z_t)/alpha_val



def sras_curve(y, p, pi_e, s):
    return pi_e + p["gamma"]*(y - p["ybar"]) + s


# Inputs: expected inflation pi_e, demand shock v, supply shock s, parameters par, 
# padding pad around ybar (width of the x-axis range around ybar), grid size n

def solve_grid(pi_e=0.02, v=0.0, s=0.0, p=par, pad=0.6, n=400):
    y = np.linspace(p["ybar"]-pad, p["ybar"]+pad, n) # grid of output values (x-axis for both curves)
    pi_ad   = ad_curve(y, p, v) # we compute the inflation values on the AD-curve
    pi_sras = sras_curve(y, p, pi_e, s) # we compute the inflation values on the SRAS-curve
    i = np.argmin(np.abs(pi_ad - pi_sras)) # find index where the two curves are closest
    return y[i], 0.5*(pi_ad[i] + pi_sras[i]), y, pi_ad, pi_sras # returns output, inflation at intersection, the output grid, and the full curves

# creating demand and supply shocks for T periods
def make_shocks(T, p, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, p["sigma_x"], T); c = rng.normal(0, p["sigma_c"], T) # innovations follow a normal distribution with given std
    v = np.zeros(T); s = np.zeros(T) # initialize shocks
    for t in range(1,T): v[t]=p["delta"]*v[t-1]+x[t]; s[t]=p["omega"]*s[t-1]+c[t] # build AR(1) shocks
    return v, s


# simulate with given rule parameters a1, a2 over shocks v,s and parameters p; uses solve_grid from above
def simulate_with_rule_grid(a1, a2, v, s, p, pad=0.6, n=400):
    P = {**p, "a1": float(a1), "a2": float(a2)}
    T=len(v); y=np.empty(T); pi=np.empty(T); pi_prev=P["pi_star"]
    for t in range(T):
        y[t], pi[t], *_ = solve_grid(pi_e=pi_prev, v=v[t], s=s[t], p=P, pad=pad, n=n)
        pi_prev = pi[t]
    return y, pi

# loss function: quadratic loss on inflation and output gap deviations (squared deviations of inflation from target and output from steady state output)
# lam is the weight on output gap deviations
def loss_quad(y, pi, p, lam=0.5):
    return float(np.sum((pi-p["pi_star"])**2 + lam*(y-p["ybar"])**2))



import numpy as np

def compute_r2(x_data, x_fit):
    """
    Compute R^2 from an OLS regression of x_data on x_fit with an intercept.

    Parameters
    ----------
    x_data : array_like
        Actual data series (dependent variable).
    x_fit : array_like
        Fitted values or model series (regressor).

    Returns
    -------
    R2 : float
        Coefficient of determination from regressing x_data on x_fit.
    a_hat : float
        Estimated intercept.
    b_hat : float
        Estimated slope.
    """
    x_data = np.asarray(x_data, dtype=float).ravel()
    x_fit  = np.asarray(x_fit,  dtype=float).ravel()

    # Design matrix with intercept
    X = np.vstack([np.ones_like(x_fit), x_fit]).T

    beta_hat, _, _, _ = np.linalg.lstsq(X, x_data, rcond=None)
    a_hat, b_hat = beta_hat

    x_pred = a_hat + b_hat * x_fit

    ss_res = np.sum((x_data - x_pred)**2)
    ss_tot = np.sum((x_data - np.mean(x_data))**2)
    R2 = 1.0 * (1.0 - ss_res / ss_tot)

    return R2, a_hat, b_hat
