import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

from sklearn import linear_model as lm

import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pick_cloud_projection import pick_cid 
import calc_radius

# Given a field, return coarse observation
def observe_coarse_field(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)

    # Normalize coarse observation
    S[S > 0] = 1
    return S

# Given a field, calculate perimeter
def calc_perimeter(Z):
    return np.sum(Z[:, 1:] != Z[:, :-1]) + \
           np.sum(Z[1:, :] != Z[:-1, :])

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    # Read horizontal slice from a cloud core
    df = pick_cid(42470, 0)

    x_width = max(df.x) - min(df.x)
    y_width = max(df.y) - min(df.y)
    xy_map = np.zeros((y_width+4, x_width+4), dtype=int)
    xy_map[df.y, df.x] = 1

    # Scaling factor based on L/R
    # r_ = calc_radius.calculate_radial_distance(df)
    r_ = calc_radius.calculate_geometric_r(df)
    sizes = np.arange(int(r_), 0, -1)

    # Calculate perimeter and area
    area = np.sum(xy_map[xy_map > 0]) * c.dx**2
    p = calc_perimeter(xy_map) * c.dx

    print("Dp =", 2*np.log10(p)/np.log10(area))

    Dp = []
    for size in sizes:
        # Coefficient for horizontal scale
        C = c.dx * size

        Z = observe_coarse_field(xy_map, size)
        area = np.sum(Z) * C**2
        p = calc_perimeter(Z) * C

        Dp.append(p)
    sizes = sizes / r_

    # Fit the successive log(sizes) with log (counts)
    model = lm.RidgeCV(fit_intercept=True)
    X = np.log10(sizes)[:, None]
    model.fit (X, np.log10(Dp))

    print(f"    {model.coef_[0] * 2}")

    #---- Plotting 
    fig = plt.figure(1, figsize=(3, 3))
    fig.clf()
    sns.set_context('paper')
    sns.set_style('ticks', 
        {
            'axes.grid': False, 
            'axes.linewidth': '0.75',
            'grid.color': '0.75',
            'grid.linestyle': u':',
            'legend.frameon': False,
        })
    plt.rc('text', usetex=True)
    plt.rc('font', family='Serif')

    ax = plt.subplot(1, 1, 1)
    plt.xlabel(r'$\log_{10}$ $L/R$')
    plt.ylabel(r'$\log_{10}$ $P$')

    label = f"P = (L/R)$^{{{2 * model.coef_[0]:.3f}/2}}$"
    plt.plot(np.log10(sizes), np.log10(Dp), 
             marker='o', lw=0.75, label=label)

    xmin, xmax = np.min(np.log10(sizes)), np.max(np.log10(sizes))
    xi = np.linspace(xmin-0.1, xmax+0.1, 50)
    y_fit = model.predict(xi[:, None])
    plt.plot(xi, y_fit)

    plt.legend()

    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=300, \
                facecolor='w', transparent=True)
