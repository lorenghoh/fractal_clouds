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
    x, y = df.x, df.y
    x_axis, y_axis = c.nx, c.ny
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
        
    x_width = max(df.x) - min(df.x)
    y_width = max(df.y) - min(df.y)
    xy_map = np.zeros((y_width+1, x_width+1), dtype=int)
    xy_map[y - min(y), x - min(x)] = 1

    # Radius estimates
    r_ = calc_radius.calculate_geometric_r(df)
    sizes = np.arange(int(r_*2), 0, -1)

    area = np.array(np.sum(xy_map) * c.dx**2)
    
    areas = []
    for k in sizes:
        C = c.dx * k
        Z = observe_coarse_field(xy_map, k)
        areas.append(np.array(area - np.sum(Z)*C**2)/area)
    
    # sizes = np.array(sizes) / r_
    if len(sizes) < 6:
        return [0], [0]
    area_ = np.ones(len(sizes)) * area
    return sizes, areas, area_

if __name__ == '__main__':
    r, a, area = [], [], []
    for time in range(0, 480, 12):
        f = f'{config["tracking"]}/clouds_00000{time:03d}.pq'
        df = pq.read_pandas(f, nthreads=16).to_pandas()

        lc = find_lc.find_largest_clouds(f)
        cids = lc.index[0:80:4]

        # Translate indices to coordinates
        df['z'] = df.coord // (c.nx * c.ny)
        xy = df.coord % (c.nx * c.ny)
        df['y'] = xy // c.ny 
        df['x'] = xy % c.nx

        # Take cloud regions and trim noise
        df = df[df.type == 0]
        group = df.groupby(['cid', 'z'], as_index=False)
        for cid in cids:
            grp = df[df.cid == cid]
        # for _, grp in group:
            if grp.shape[0] < 6:
                continue
            try:
                r_, a_, area_ = find_shell_area_fraction(grp)
            except:
                continue
            r.append(r_)
            a.append(a_)
            area.append(area_)
    # def group_shell(df):
    #     r_, a_ = find_shell_area_fraction(df)
    #     return pd.DataFrame({'r_': r_,
    #                          'a_': a_})

    # with Parallel(n_jobs=16) as Pr:
    #     result = Pr(delayed(group_shell)
    #                 (grouped) for _, grouped in group)
    # df = pd.concat(result, ignore_index=True)

    #---- Plotting 
    fig = plt.figure(1, figsize=(4, 4))
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

    r = np.concatenate(r).ravel() * c.dx
    a = np.concatenate(a).ravel()
    area = np.concatenate(area).ravel() / 1e6
    m_ = (r > 0) & (a > 0)

    cmap = sns.cubehelix_palette(start=.3, rot=-.4, as_cmap=True)
    sc = plt.scatter(r[m_], a[m_], c=area[m_], s=5, cmap=cmap)
    cb = plt.colorbar(sc, label=r'Area [km$^2$]')

    # plt.xlim([0, 1.25])
    plt.xlim([0, 800])

    plt.xlabel(r'$l$ [m]')
    plt.ylabel(r'$\mathcal{A}_s/\mathcal{A}$')

    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=300, \
                facecolor='w', transparent=True)
