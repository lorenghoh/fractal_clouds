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
    if (max(x) - min(x)) >= x_axis // 2:
        x_off = x_axis - np.min(x[(x > x_axis // 2)])
        
        # Shift x-coordinates
        x = x + x_off
        x[x >= x_axis] = x[x >= x_axis] - x_axis

    if (max(y) - min(y)) >= y_axis // 2:
        y_off = y_axis - np.min(y[(y > y_axis // 2)])

        # Shift y-coordinates
        y = y + y_off
        y[y >= y_axis] = y[y >= y_axis] - y_axis
        
    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    xy_map = np.zeros((y_width+4, x_width+4), dtype=int)
    xy_map[y - min(y), x - min(x)] = 1

    # Radius estimates
    r_ = calc_radius.calculate_geometric_r(df)
    l_set = np.arange(int(r_*2), 0, -1)

    area = np.array(np.sum(xy_map) * c.dx**2)
    
    xs, ws = [], []
    for k in l_set:
        C = c.dx * k
        Z = observe_coarse_field(xy_map, k)

        xs.append(np.array(np.sum(Z)*C**2) / area)
        ws.append(area / 1e6)
    
    # ls = np.array(l_set) / r_
    # Use filter for l
    ls = np.array(l_set)
    # if len(ls) < 6:
    #     return [0], [0], [0]
    return ls, xs, ws

if __name__ == '__main__':
    r, a, w = [], [], []
    df_ = pd.DataFrame()
    for time in range(0, 540, 30):
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
            r_, a_, w_ = find_shell_area_fraction(df)
            return pd.DataFrame({'r_': r_,
                                 'a_': a_,
                                 'w_': w_})

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

    r, a, w = df_.r_, df_.a_, df_.w_
    m_ = (r > 0) & (r < 32) & (a > 0)
    
    # Colorbar scheme used for Scattergram
    def hist_weight(H, w, x, y, xi, yi, bins=len(np.unique(r[m_]))+5):
        W = np.zeros_like(H)
        for i in range(bins):
            for j in range(bins):
                if np.isfinite(H[j, i]) & (H[j, i] > 0):
                    _m = (x >= xi[i]) & (x <= xi[i+1])
                    _m = _m & (y >= yi[j]) & (y <= yi[j+1])
                    W[j, i] = np.nanmean(w[_m])
        return W

    cmap = sns.cubehelix_palette(start=1.2, hue=1, light=1, 
                                 rot=-1.05, as_cmap=True)
    H, xi, yi = np.histogram2d(r[m_], a[m_], bins=len(np.unique(r[m_]))+5)
    H = H.T
    W = hist_weight(H, w[m_], r[m_], a[m_], xi, yi)

    xii, yii = np.meshgrid(xi[1:], yi[1:])
    print(xii[0])
    sc = plt.scatter(xii, yii, s=15, c=W, cmap=cmap)

    cb = plt.colorbar(sc, label=r'Area [km$^2$]')
    
    plt.xlim([0, 32])
    plt.xticks(np.arange(0, 32, 4), np.arange(0, 32, 4) * c.dx)

    plt.xlabel(r'$l$ [m]')
    plt.ylabel(r'$\mathcal{A}_s/\mathcal{A}$')

    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=300, \
                facecolor='w', transparent=True)
