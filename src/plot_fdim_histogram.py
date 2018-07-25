import os, glob, warnings
import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from load_config import c, config

if __name__ == '__main__':
    filelist = sorted(glob.glob(f"../pq/fdim_dump_*.pq"))
    df = pq.ParquetDataset(filelist).read(nthreads=16).to_pandas()
    desc = df.describe().squeeze()

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
    plt.rc('font', family='Serif')

    ax = plt.subplot(1, 1, 1)
    plt.xlabel(r'$\mathcal{D}_\mathrm{box}$')
    plt.ylabel(r'Frequency')

    cmap = sns.cubehelix_palette(start=1.2, hue=1, \
                                 light=1, rot=-1.05, as_cmap=True)
    sns.distplot(df, ax=ax, bins=20)

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
    ax.text(0, 3.5, box_text, fontsize=10, va='top', 
            bbox=dict(boxstyle='round, pad=0.5', fc='w'))

    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)