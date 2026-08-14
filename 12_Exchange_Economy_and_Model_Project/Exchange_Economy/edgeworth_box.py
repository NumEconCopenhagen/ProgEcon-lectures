""" the Edgeworth box from lecture 10

The plotting code for the Edgeworth box, kept apart from the model itself in
[exchange.py](exchange.py): `edgeworth_box` knows nothing about the model at all,
and the two indifference-curve functions only need `utility_A` and `utility_B`.

"""

import numpy as np
import matplotlib.pyplot as plt

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
    flip = (lambda x: 1-x, lambda x: 1-x) # the same function both ways
    ax.secondary_xaxis('top',functions=flip).set_xlabel('$x_1^B$')
    ax.secondary_yaxis('right',functions=flip).set_ylabel('$x_2^B$')

    # d. a bit of air around the box
    ax.set_xlim(-0.05,1.05)
    ax.set_ylim(-0.05,1.05)
    ax.set_aspect('equal') # a square is only a square if the axes have the same scale

    return fig,ax

def plot_indifference_A(ax,model,x1A,x2A,**kwargs):
    """ plot A's indifference curve through the point (x1A,x2A) """

    uA = model.utility_A(x1A,x2A)

    x1_grid = np.linspace(0.001,0.999,1000)
    x2_grid = (uA/x1_grid**model.alpha)**(1/(1-model.alpha)) # solved for x2

    I = (x2_grid > 0) & (x2_grid < 1) # only what is inside the box
    ax.plot(x1_grid[I],x2_grid[I],**kwargs)

def plot_indifference_B(ax,model,x1B,x2B,**kwargs):
    """ plot B's indifference curve through the point (x1B,x2B), flipped into A's coordinates """

    uB = model.utility_B(x1B,x2B)

    x1B_grid = np.linspace(0.001,0.999,1000)
    x2B_grid = (uB/x1B_grid**model.beta)**(1/(1-model.beta)) # solved for x2

    x1_grid,x2_grid = 1-x1B_grid,1-x2B_grid # flip into A's coordinates

    I = (x2_grid > 0) & (x2_grid < 1)
    ax.plot(x1_grid[I],x2_grid[I],**kwargs)
