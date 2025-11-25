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