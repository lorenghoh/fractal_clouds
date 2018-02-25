import os, glob, warnings
import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pick_cloud_projection import pick_cid 

def count_box(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)

    # We count non-empty (0) and non-full boxes (k*k)
    return len(np.where((S > 0) & (S < k*k))[0])

def adjust_coordinates(x, y):
    x_axis, y_axis = 256, 256
    if (max(x) - min(x)) > x_axis // 2:
        x_off = x_axis - np.min(x[(x > x_axis // 2)])

        # Shift x-coordinates
        x = x + x_off
        x[x >= x_axis] = x[x >= x_axis] - x_axis

    if (max(y) - min(y)) > y_axis // 2:
        y_off = y_axis - np.min(y[(y > y_axis // 2)])

        # Shift y-coordinates
        y = y + y_off
        y[y >= y_axis] = y[y >= y_axis] - y_axis

    return x - min(x) + 1, y - min(y) + 1

def calculate_fdim(df):
    x, y = adjust_coordinates(df.x, df.y)

    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    xy_map = np.zeros((y_width+4, x_width+4), dtype=int)
    xy_map[y, x] = 1

    # Leave only the perimeter of the cloud 
    xy_temp = np.roll(xy_map, 1, axis=0) \
            + np.roll(xy_map, -1, axis=0) \
            + np.roll(xy_map, 1, axis=1) \
            + np.roll(xy_map, -1, axis=1)
    xy_map[xy_temp == 4] = 0

    # Build successive box sizes (from 2**n down to 2**1)
    p = min(xy_map.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(count_box(xy_map, size))
    
    # Fit the successive log(sizes) with log (counts)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            c = np.polyfit(np.log(sizes), np.log(counts), 1)
            return c, sizes, counts
        except:
            return [0, 0], 0, 0

if __name__ == '__main__':
    path = '/nodessd/loh/repos/tracking_parq'
    filelist = sorted(glob.glob(f'{path}/clouds_*.pq'))
    df = pq.ParquetDataset(filelist).read(nthreads=6).to_pandas()

    # Take 30 largest cloud cores in the dataset
    largest_clouds = df.cid.value_counts()[:600].index
    df = df[df.cid.isin(largest_clouds)]
    df = df[df.type == 0]

    # Translate indices to coordinates
    df['z'] = df.coord // (256 * 256)
    xy = df.coord % (256 * 256)
    df['y'] = xy // 256 
    df['x'] = xy % 256

    # Project the 3D cloud onto surface 
    df_ = df.drop_duplicates(subset=['y', 'x'], keep='first')

    def calc_fdim_to_df(df):
        c, _, _ = calculate_fdim(df)
        return pd.Series({'fdim': -c[0]})

    group = df.groupby(['cid'], as_index=False)
    df_fdim = group.apply(calc_fdim_to_df)

    df_fdim = df_fdim[df_fdim.fdim > 0]
    desc = df_fdim.describe().squeeze()
    print(desc) # Print statistics

    #---- Plotting 
    fig = plt.figure(1, figsize=(4.5, 3))
    fig.clf()
    sns.set_context('paper')
    sns.set_style('ticks', 
        {
            'axes.grid': False, 
            'axes.linewidth': '0.75',
            'grid.color': '0.75',
            'grid.linestyle': u':',
            'legend.frameon': True,
        })
    plt.rc('text', usetex=True)
    plt.rc('font', family='Helvetica')

    ax = plt.subplot(1, 1, 1)
    plt.xlabel(r'$\mathcal{D}_\mathrm{box}$')
    plt.ylabel(r'Frequency')

    cmap = sns.cubehelix_palette(start=1.2, hue=1, \
                                 light=1, rot=-1.05, as_cmap=True)
    sns.distplot(df_fdim, ax=ax, bins=20)

    # Normal distribution given histogram statistics
    xi = np.linspace(0, 2.2, 100)
    mu_ = desc['mean']
    sig_ = desc['std']
    plt.plot(xi, (2 * np.pi * sig_**2)**(-0.5) * \
             np.exp(-(xi - mu_)**2 / (2 * sig_**2)))

    # Text box with distribution specs
    box_text = "Count: {:,} \n".format(int(desc['count'])) \
                + f"Mean: {mu_:.3f} \n " \
                + f"Std: {sig_:.3f}"
    ax.text(0, 3.6, box_text, fontsize=10, va='top', 
            bbox=dict(boxstyle='round, pad=0.5', fc='w'))

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)