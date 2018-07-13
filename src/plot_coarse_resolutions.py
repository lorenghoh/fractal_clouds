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
    path = '/nodessd/loh/repos/tracking_parq'
    df = pq.read_table(f'{path}/clouds_00000121.pq', nthreads=6).to_pandas()
    df = df[(df.cid == 4563) & (df.type == 0)]

    # Calculate z index from coordinates
    df['z'] = df.coord // (256 * 256)

    k_sample = df.z.value_counts().index[0]
    df = df[(df.z == k_sample)]

    # From there, take the xy indices
    xy = df.coord % (256 * 256)
    df['y'] = xy // 256 
    df['x'] = xy % 256

    # Drop duplicates 
    df = df.drop_duplicates(subset=['y', 'x'], keep='first')

    x = df.x.values
    y = df.y.values

    # Map cloud core onto the BOMEX domain and adjust
    xy_map = np.zeros((256, 256), dtype=int)
    xy_map[y, x] = 1

    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    print(f"\nCorresponding sub-domain size: {y_width}x{x_width}")
    print( "Adjusted coordinates: " \
          f"({min(y)}, {max(y)}), ({min(x)}, {max(x)})")

    # Map the projection onto a new 2D array (+4 padding)
    Z = np.zeros((y_width+4, x_width+4), dtype=int)
    x_sub = x - min(x) + 1
    y_sub = y - min(y) + 1
    Z[y_sub, x_sub] = 1

    # Unmodified observed cloud field
    Z_base = np.copy(Z)

    Z = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], 3), axis=0),
                           np.arange(0, Z.shape[1], 3), axis=1)

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
    plt.rc('font', family='Helvetica')

    cmap = sns.cubehelix_palette(start=1.2, hue=1, \
                                 light=1, rot=-1.05, as_cmap=True)

    ax = plt.subplot(1, 1, 1)
    xi = np.arange(Z.shape[1])
    yi = np.arange(Z.shape[0])

    im = plt.pcolormesh(xi, yi, Z, cmap=cmap, lw=0.5)

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)