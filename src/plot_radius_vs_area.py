import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import calc_radius
import find_largest_clouds as find_lc

from scipy.stats import gaussian_kde
from joblib import Parallel, delayed

import load_config
c, config = load_config.c, load_config.config

# Given a field, return coarse observation
def observe_coarse_field(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)

    # Normalize coarse observation
    thold = k**2 * 0.98
    S[S < thold] = 0
    S[S >= thold] = 1
    return S

def calc_perimeter(Z):
    return np.sum(Z[:, 1:] != Z[:, :-1]) + \
           np.sum(Z[1:, :] != Z[:-1, :])

def find_shell_area_fraction(df):
    x, y = df.x, df.y
    x_axis, y_axis = c.nx, c.ny
    if (max(x) - min(x)) >= x_axis // 2:
        x_off = x_axis - np.min(x[(x >= x_axis // 2)])
        
        # Shift x-coordinates
        x = x + x_off
        x[x >= x_axis] = x[x >= x_axis] - x_axis

    if (max(y) - min(y)) >= y_axis // 2:
        y_off = y_axis - np.min(y[(y >= y_axis // 2)])

        # Shift y-coordinates
        y = y + y_off
        y[y >= y_axis] = y[y >= y_axis] - y_axis
        
    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    xy_map = np.zeros((y_width+4, x_width+4), dtype=int)
    xy_map[y - min(y), x - min(x)] = 1

    # Radius and area estimates
    r_d = calc_radius.calculate_radial_distance(xy_map)
    r_g = calc_radius.calculate_geometric_r(xy_map)
    area = np.array(np.sum(xy_map) * c.dx**2)
    
    # Assume core size of 0.3 R
    # Measure core / shell ratio
    k = np.ceil(r_g * 0.3).astype(np.int_)
    Z = observe_coarse_field(xy_map, k)

    shell_area = (area - np.sum(Z) * (k * c.dx)**2) / 1e6
    perimeter = calc_perimeter(xy_map) * c.dx / 1e3
    if k < 3:
        return 0, 0

    return perimeter, shell_area

if __name__ == '__main__':
    r, a, w = [], [], []
    df_ = pd.DataFrame()
    for time in range(0, 540, 5):
        f = f'{config["tracking"]}/clouds_00000{time:03d}.pq'
        df = pq.read_pandas(f, nthreads=16).to_pandas()

        lc = find_lc.find_largest_clouds(f)
        cids = lc.index[:128]

        # Translate indices to coordinates
        df['z'] = df.coord // (c.nx * c.ny)
        xy = df.coord % (c.nx * c.ny)
        df['y'] = xy // c.ny 
        df['x'] = xy % c.nx

        # Take cloud regions and trim noise
        df = df[df.cid.isin(cids) & (df.type == 0)]
        group = df.groupby(['cid', 'z'])
        def group_shell(df):
            df_x, df_y, = find_shell_area_fraction(df)
            return pd.DataFrame({'x': [df_x],
                                 'y': [df_y]})

        with Parallel(n_jobs=16) as Pr:
            result = Pr(delayed(group_shell)(grouped) for _, grouped in group)
            df_r = pd.concat(result, ignore_index=True)
        df_ = pd.concat([df_, df_r], ignore_index=True)

    #---- Plotting 
    fig = plt.figure(1, figsize=(4.5, 4))
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

    x, y = df_.x, df_.y
    m_ = (x > 0) & (y > 0)

    x = x[m_]
    y = y[m_]

    cmap = sns.cubehelix_palette(start=1.2, hue=1, light=1, rot=-1.05, as_cmap=True)
    H, xi, yi = np.histogram2d(x, y, bins=60)
    im = plt.pcolormesh(xi, yi, np.log10(H.T), vmin=.1, vmax=3, cmap=cmap, \
        alpha=.9, edgecolor='0.9', linewidths = (0,),)
    plt.colorbar(label=r'$\log_{10}$ Frequency')

    plt.xlabel(r'Cloud Perimeter $P$ [km]')
    plt.ylabel(r'Shell Area $\mathcal{A}_s(l = 0.3 R)$ [km$^2$]')

    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=300, \
                facecolor='w', transparent=True)
