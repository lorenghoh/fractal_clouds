import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pick_cloud_projection import pick_cid 
import calc_radius
import calc_pdim

# Given a field, return coarse observation
def observe_coarse_field(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)

    # Normalize coarse observation
    thold = k**2 * 0.95
    S[S < thold] = 0
    S[S >= thold] = 1
    return S

if __name__ == '__main__':
    df = pick_cid(4563, 4)

    x_width = max(df.x) - min(df.x)
    y_width = max(df.y) - min(df.y)
    xy_map = np.zeros((y_width+4, x_width+4), dtype=int)
    xy_map[df.y, df.x] = 1

    # Radius estimates 
    r_g = calc_radius.calculate_radial_distance(df)
    r_d = calc_radius.calculate_geometric_r(df)
    sizes = np.arange(int(r_g), 0, -1)

    # Resolution
    dx = 25

    size, area = dx, np.array(np.sum(xy_map) * dx**2)

    areas = []
    for size in sizes:
        dxx = dx * size
        Z = observe_coarse_field(xy_map, size)
        areas.append(np.array(area - np.sum(Z) * dxx**2)/area)

    sizes = np.array(sizes)/r_g
    m_ = (sizes > 0.4)
    c1 = np.polyfit(sizes[m_], np.array(areas)[m_], 1)
    c2 = np.polyfit(sizes[~m_], np.array(areas)[~m_], 1)
    c2 = np.polyfit(sizes[~m_], np.array(areas)[~m_], 1)

    #---- Plotting 
    fig = plt.figure(1, figsize=(3, 3))
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

    cmap = sns.cubehelix_palette(start=1.2, hue=1, \
                                 light=1, rot=-1.05, as_cmap=True)

    ax = plt.subplot(1, 1, 1)
    plt.xlabel(r'$S$')
    plt.ylabel(r'$\mathcal{A}_\mathrm{s}/\mathcal{A}$')

    xi = np.linspace(min(sizes)-0.2, max(sizes)+0.2, 50)
    plt.plot(xi, c1[0]*xi+c1[1], lw=0.9, label="S $>$ 0.4")
    xi = np.linspace(min(sizes[~m_])-0.1, max(sizes[~m_])+0.1, 50)
    plt.plot(xi, c2[0]*xi+c2[1], lw=0.9, label="S $\leq$ 0.4")

    plt.plot(sizes, areas, marker='o', lw=0.75)

    plt.legend(loc=4)

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=300, \
                facecolor='w', transparent=True)
