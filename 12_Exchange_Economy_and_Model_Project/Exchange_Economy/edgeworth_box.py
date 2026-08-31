""" the Edgeworth box from lecture 10

The plotting code for the Edgeworth box, kept apart from the model itself in
[exchange.py](exchange.py): `edgeworth_box` knows nothing about the model at all,
and the two indifference-curve functions only need `utility_A` and `utility_B`.

"""

import numpy as np
import matplotlib.pyplot as plt

#######
# Box #
#######

def edgeworth_box(figsize=(7,7)):
    """ an empty Edgeworth box, everything is measured in A's coordinates

    Args:

        figsize (tuple): size of the figure

    Returns:

        (Figure,Axes): the figure and the axes to draw in

    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)

    # a. the sides of the box
    ax.plot([0,1,1,0,0],[0,0,1,1,0],lw=2,color='black')

    # b. A's axes, read from the lower left corner
    ax.set_xlabel('$x_1^A$')
    ax.set_ylabel('$x_2^A$')

    # c. B's axes, the same box read from the upper right corner
    flip = (lambda x: 1-x, lambda y: 1-y) # the same function both ways
    ax.secondary_xaxis('top',functions=flip).set_xlabel('$x_1^B$')
    ax.secondary_yaxis('right',functions=flip).set_ylabel('$x_2^B$')

    # d. a bit of air around the box
    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(-0.05,1.05)
    ax.set_aspect('equal') # a square is only a square if the axes have the same scale

    return fig,ax

#####
# A #
#####

def get_indifference_A(model,x1A,x2A,x1_grid=None):
    """ return the points of A's indifference curve through the point (x1A,x2A) """

    uA = model.utility_A(x1A,x2A)

    if x1_grid is None: x1_grid = np.linspace(0.001,0.999,1000)
    x2_grid = (uA/x1_grid**model.alpha)**(1/(1-model.alpha)) # solved for x2

    return x1_grid,x2_grid

def plot_indifference_A(ax,model,x1A,x2A,**kwargs):
    """ plot A's indifference curve through the point (x1A,x2A) """

    x1_grid,x2_grid = get_indifference_A(model,x1A,x2A)
    I = (x2_grid > 0) & (x2_grid < 1) # only what is inside the box
    ax.plot(x1_grid[I],x2_grid[I],**kwargs)

#####
# B #
#####

def get_indifference_B(model,x1B,x2B,x1_grid=None):
    """ return the points of B's indifference curve through the point (x1B,x2B), flipped into A's coordinates """

    uB = model.utility_B(x1B,x2B)

    if x1_grid is None: x1_grid = np.linspace(0.001,0.999,1000)
    x2_grid = (uB/x1_grid**model.beta)**(1/(1-model.beta)) # solved for x2

    return x1_grid,x2_grid

def plot_indifference_B(ax,model,x1B,x2B,**kwargs):
    """ plot B's indifference curve through the point (x1B,x2B), flipped into A's coordinates """

    x1_grid,x2_grid = get_indifference_B(model,x1B,x2B)

    x1_grid,x2_grid = 1-x1_grid,1-x2_grid # flip into A's coordinates
    I = (x2_grid > 0) & (x2_grid < 1) # only what is inside the box

    ax.plot(x1_grid[I],x2_grid[I],**kwargs)

###############
# Improvement #
###############

def plot_improvement_set(ax,model,x1A,x2A,**kwargs):
    """ plot the set of allocations that make both agents better off than at the endowment """

    x1_grid = np.linspace(0.001,0.999,1000)
    _,x2A_grid = get_indifference_A(model,x1A,x2A,x1_grid=x1_grid)
    _,x2B_grid = get_indifference_B(model,1-x1A,1-x2A,x1_grid=1-x1_grid)

    lower = x2A_grid # A must be above this
    upper = 1-x2B_grid # B must be below this

    I = (upper > lower) & (lower > 0) & (lower < 1)
    ax.fill_between(x1_grid[I],lower[I],upper[I],color='black',alpha=0.12,label='both are better off here')

