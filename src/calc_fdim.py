import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pick_cloud_projection import pick_cid
import calc_radius

def count_box(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)

    # We count non-empty (0) and non-full boxes (k*k)
    return len(np.where((S > 0) & (S < k*k))[0])

def calculate_fdim(df):
    x_width = max(df.x) - min(df.x)
    y_width = max(df.y) - min(df.y)
    xy_map = np.zeros((y_width+4, x_width+4), dtype=int)
    xy_map[df.y, df.x] = 1

    # Leave only the perimeter of the cloud 
    xy_temp = np.roll(xy_map, 1, axis=0) \
            + np.roll(xy_map, -1, axis=0) \
            + np.roll(xy_map, 1, axis=1) \
            + np.roll(xy_map, -1, axis=1)
    xy_map[xy_temp == 4] = 0

    # Build successive box sizes (from 2**n down to 2**1)
    # p = min(xy_map.shape)
    # n = 2**np.floor(np.log(p)/np.log(2))
    # n = int(np.log(n)/np.log(2))
    # sizes = 2**np.arange(n, 1, -1)

    # Scaling factor based on L/R
    r_g = calc_radius.calculate_radial_distance(df)
    r_d = calc_radius.calculate_geometric_r(df)
    sizes = np.arange(int(r_d), 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(count_box(xy_map, size))
    
    sizes = sizes / r_d
    # Fit the successive log(sizes) with log (counts)
    c = np.polyfit(np.log(sizes), np.log(counts), 1)
    print(-c[0])

    return c, sizes, counts

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    # Read cloud core projection image
    df = pick_cid(4563, 0)

    c, sizes, counts = calculate_fdim(df)

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
    plt.rc('font', family='Helvetica')

    cmap = sns.cubehelix_palette(start=1.2, hue=1, \
                                 light=1, rot=-1.05, as_cmap=True)

    ax = plt.subplot(1, 1, 1)
    plt.xlabel(r'$\log_{10}$ $\eta$')
    plt.ylabel(r'$\log_{10}$ N($\eta$)')

    plt.plot(np.log(sizes), np.log(counts), marker='o', lw=0.75)
    xi = np.linspace(-2.25, 0.2, 50)
    label = r"$\mathcal{D}_\mathrm{box}$ = " + f"{-c[0]:.3f}"
    plt.plot(xi, c[0]*xi+c[1], 'k-', lw=0.9, label=label)

    plt.legend()

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=300, \
                facecolor='w', transparent=True)