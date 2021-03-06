import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pick_cloud_projection import pick_cid 

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    # Read horizontal slice from a cloud core
    xy_map, x, y = pick_cid(4563, 0)

    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    print(f"\nCorresponding sub-domain size: {y_width}x{x_width}")
    print( "Adjusted coordinates: " \
          f"({min(y)}, {max(y)}), ({min(x)}, {max(x)})")

    # Map the projection onto a new 2D array (+4 padding)
    xy_map_sub = np.zeros((y_width+4, x_width+4), dtype=int)
    x_sub = x - min(x) + 1
    y_sub = y - min(y) + 1
    xy_map_sub[y_sub, x_sub] = 1

    print(xy_map_sub.shape)

    # Leave only the perimeter of the cloud 
    xy_temp = np.roll(xy_map_sub, 1, axis=0) \
            + np.roll(xy_map_sub, -1, axis=0) \
            + np.roll(xy_map_sub, 1, axis=1) \
            + np.roll(xy_map_sub, -1, axis=1)
    xy_map_per = np.array(xy_map_sub, copy=True)
    xy_map_per[xy_temp >= 4] = 0

    #---- Plotting 
    fig = plt.figure(1, figsize=(4, 6))
    fig.clf()
    sns.set_context('paper')
    sns.set_style('ticks', 
        {
            'axes.grid': True, 
            'axes.linewidth': '0.75',
            'grid.color': '0.75',
            'grid.linestyle': u':',
            'legend.frameon': True,
        })
    plt.rc('text', usetex=True)
    plt.rc('font', family='Helvetica')

    cmap = sns.cubehelix_palette(start=1.2, hue=1, \
                                 light=1, rot=-1.05, as_cmap=True)

    ax = plt.subplot(2, 1, 1)

    xi = np.arange(x_width+4)
    yi = np.arange(y_width+4)
    im = plt.pcolormesh(xi, yi, xy_map_sub, cmap=cmap, lw=0.5)

    ax = plt.subplot(2, 1, 2)
    im = plt.pcolormesh(xi, yi, xy_map_per, cmap=cmap, lw=0.5)

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)