import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import calc_radius

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

def find_shell_area_fraction(df):
    x_width = max(df.x) - min(df.x)
    y_width = max(df.y) - min(df.y)
    xy_map = np.zeros((y_width+1, x_width+1), dtype=int)
    xy_map[df.y, df.x] = 1

    # Radius estimates
    r_ = calc_radius.calculate_geometric_r(df)
    sizes = np.arange(int(r_), 0, -1)

    size, area = c.dx, np.array(np.sum(xy_map) * c.dx**2)

    areas = []
    for size in sizes:
        dxx = c.dx * size
        Z = observe_coarse_field(xy_map, size)
        areas.append(np.array(area - np.sum(Z) * dxx**2)/area)

    sizes = np.array(sizes) / r_
    return sizes, areas

if __name__ == '__main__':
    f = f'{config["tracking"]}/clouds_00000121.pq'
    df = pq.read_pandas(f, nthreads=16).to_pandas()

    # Translate indices to coordinates
    df['z'] = df.coord // (c.nx * c.ny)
    xy = df.coord % (c.nx * c.ny)
    df['y'] = xy // c.ny 
    df['x'] = xy % c.nx

    # Take cloud regions and trim noise
    df = df[df.type == 0]
    group = df.groupby(['cid', 'z'], as_index=False)

    def group_shell(df):
        s_, a_ = find_shell_area_fraction(df)
        return pd.DataFrame({'size': [s_],
                             'area': [a_]})

    with Parallel(n_jobs=16) as Pr:
        result = Pr(delayed(group_shell)
                    (grouped) for _, grouped in group)
    df = pd.concat(result, ignore_index=True)

    #---- Plotting 
    fig = plt.figure(1, figsize=(8, 6))
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

    plt.scatter(df.size, df.area)

    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=300, \
                facecolor='w', transparent=True)
