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
    xy_map = np.zeros((y_width+1, x_width+1), dtype=int)
    xy_map[y - min(y), x - min(x)] = 1

    # Radius estimates
    r_ = calc_radius.calculate_geometric_r(df)
    l_set = np.arange(int(r_*2), 0, -1)

    area = np.array(np.sum(xy_map) * c.dx**2)
    
    xs, ws = [], []
    for k in l_set:
        C = c.dx * k
        Z = observe_coarse_field(xy_map, k)

        xs.append((area - np.array(np.sum(Z)*C**2))/area)
        ws.append(area / 1e6)
    
    ls = np.array(l_set) / r_
    if len(ls) < 6:
        return [0], [0], [0]
    return ls, xs, ws

if __name__ == '__main__':
    r, a, w = [], [], []
    for time in range(0, 540, 15):
        f = f'{config["tracking"]}/clouds_00000{time:03d}.pq'
        df = pq.read_pandas(f, nthreads=16).to_pandas()

        lc = find_lc.find_largest_clouds(f)
        cids = lc.index[0:128]

        # Translate indices to coordinates
        df['z'] = df.coord // (c.nx * c.ny)
        xy = df.coord % (c.nx * c.ny)
        df['y'] = xy // c.ny 
        df['x'] = xy % c.nx

        # Take cloud regions and trim noise
        df = df[df.type == 0]
        for cid in cids:
            grp = df[df.cid == cid]
            if grp.shape[0] < 6:
                continue
            try:
                r_, a_, w_ = find_shell_area_fraction(grp)
            except:
                continue
            r.append(r_)
            a.append(a_)
            w.append(w_)

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

    r = np.concatenate(r).ravel()
    a = np.concatenate(a).ravel()
    w = np.concatenate(w).ravel()
    m_ = (r > 0) & (a > 0)

    # Scattergram
    def hist_weight(H, w, x, y, xi, yi, bins=40):
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
    H, xi, yi = np.histogram2d(r[m_], a[m_], bins=40)
    H = H.T
    H[H < 5] = np.nan
    W = hist_weight(H, w[m_], r[m_], a[m_], xi, yi)

    xii, yii = np.meshgrid(xi[1:], yi[1:])
    sc = plt.scatter(xii, yii, s=15, c=W, cmap=cmap)

    cb = plt.colorbar(sc, label=r'Area [km$^2$]')

    plt.xlabel(r'$S(l)$')
    plt.ylabel(r'$\mathcal{A}_s/\mathcal{A}$')

    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=300, \
                facecolor='w', transparent=True)
