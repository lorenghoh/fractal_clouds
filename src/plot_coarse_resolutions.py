import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pick_cloud_projection import pick_cid
import load_config
c, config = load_config.c, load_config.config

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    # Read horizontal slice from a cloud core
    f = f'{config["tracking"]}/clouds_00000121.pq'
    df = pq.read_pandas(f, nthreads=16).to_pandas()
    df = df[(df.cid == 11281) & (df.type == 0)]

    # Calculate z index from coordinates
    df['z'] = df.coord // (c.nx * c.ny)

    k_sample = df.z.value_counts().index[0]
    df = df[(df.z == k_sample)]

    # From there, take the xy indices
    xy = df.coord % (c.nx * c.ny)
    df['y'] = xy // c.ny 
    df['x'] = xy % c.nx

    # Drop duplicates 
    df = df.drop_duplicates(subset=['y', 'x'], keep='first')

    x = df.x.values
    y = df.y.values

    # Map cloud core onto the BOMEX domain and adjust
    xy_map = np.zeros((c.nx, c.ny), dtype=int)
    xy_map[y, x] = 1

    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    print(f"\nCorresponding sub-domain size: {y_width}x{x_width}")
    print( "Adjusted coordinates: " \
          f"({min(y)}, {max(y)}), ({min(x)}, {max(x)})")

    # Map the projection onto a new 2D array (+1 padding)
    Z = np.zeros((y_width+1, x_width+1), dtype=int)
    x_sub = x - min(x)
    y_sub = y - min(y)
    Z[y_sub, x_sub] = 1

    Z_base = np.copy(Z)
    coarse_maps = []
    for i in [1, 2, 3, 6]:
        # Unmodified observed cloud field
        Z = np.add.reduceat(
            np.add.reduceat(Z_base, np.arange(0, Z_base.shape[0], i), axis=0),
                            np.arange(0, Z_base.shape[1], i), axis=1)
        # Re-adjust coarse observation
        S = np.zeros((Z.shape[0]+3, Z.shape[1]+3))
        S[1:-2, 1:-2] = Z[:]
        S[S > 0] = 1

        coarse_maps.append(S)

    #---- Plotting 
    fig = plt.figure(1, figsize=(5, 4))
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
    plt.rc('font', family='Serif')

    cmap = sns.cubehelix_palette(start=1.2, hue=1, \
                                 light=1, rot=-1.05, as_cmap=True)

    for i, aid in zip([1, 2, 3, 6], [1, 2, 3, 4]):
        ax = plt.subplot(2, 2, aid)
        if aid <= 1:
            xi = np.arange(coarse_maps[aid-1].shape[1]) * c.dx * i / 1e3
            yi = np.arange(coarse_maps[aid-1].shape[0]) * c.dx * i / 1e3
        else:
            xi = np.arange(coarse_maps[aid-1].shape[1]) * c.dx * i / 1e3
            yi = np.arange(coarse_maps[aid-1].shape[0]) * c.dx * i / 1e3

        plt.title(f'$l = {i * c.dx}$ m')
        im = plt.pcolormesh(xi, yi, coarse_maps[aid-1], cmap=cmap, lw=0.5)

    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)