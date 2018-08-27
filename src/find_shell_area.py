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
import find_largest_clouds as find_lc

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
    xy_map = np.zeros((y_width+4, x_width+4), dtype=int)
    xy_map[df.y, df.x] = 1

    # Radius estimates 
    # r_ = calc_radius.calculate_radial_distance(df)
    r_ = calc_radius.calculate_geometric_r(df)
    sizes = np.arange(int(r_), 0, -1)

    # Resolution
    dx = 25

    size, area = dx, np.array(np.sum(xy_map) * dx**2)

    areas = []
    for size in sizes:
        dxx = dx * size
        Z = observe_coarse_field(xy_map, size)
        areas.append(np.array(area - np.sum(Z) * dxx**2)/area)

    sizes = np.array(sizes) / r_
    return sizes, areas

if __name__ == '__main__':
    r_, sizes, areas = [], [], []
    c1, c2 = [], []

    filename = f'{config["tracking"]}/clouds_00000121.pq'
    lc = find_lc.find_largest_clouds(filename)
    cids = lc.index[0:120:10]

    max_index = 6
    for i in range(max_index):
        df = pick_cid(cids[i], 0)

        s_, a_ = find_shell_area_fraction(df)
        r_.append(np.sqrt(df.shape[0]/np.pi))
        sizes.append(s_)
        areas.append(a_)

        m_ = (s_ > 0.4)
        c_ = np.polyfit(s_[m_], np.array(a_)[m_], 1)
        c1.append(c_)
        c_ = np.polyfit(s_[~m_], np.array(a_)[~m_], 1)
        c2.append(c_)

    #---- Plotting 
    # fig = plt.figure(1, figsize=(10, 6))
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

    pal = sns.color_palette()

    for i in range(max_index):
        ax = plt.subplot(2, 3, i+1)
        if i >= 3:
            plt.xlabel(r'$S$')

        if i in [0, 3]:
            plt.ylabel(r'$\mathcal{A}_\mathrm{s}/\mathcal{A}$')

        plt.title(rf'r $\approx$ {r_[i]*c.dx:.2f} [m]')
        plt.plot(sizes[i], areas[i], marker='o', c='k', lw=1.25)

        m_ = (sizes[i] > 0.4)

        xi = np.linspace(min(sizes[i])-0.2, max(sizes[i])+0.2, 50)
        plt.plot(xi, c1[i][0]*xi+c1[i][1], '--', 
                 lw=0.75, c=pal[0], label=f"S $>$ 0.4")
        xi = np.linspace(min(sizes[i][~m_])-0.1, max(sizes[i][~m_])+0.1, 50)
        plt.plot(xi, c2[i][0]*xi+c2[i][1], '--', lw=0.75,
                 c=pal[1], label=f"S $\leq$ 0.4")

        plt.legend(loc=4)

    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=300, \
                facecolor='w', transparent=True)
