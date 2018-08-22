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

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    # Read horizontal slice from a cloud core
    df = pick_cid(3621, 0)

    x_width = max(df.x) - min(df.x)
    y_width = max(df.y) - min(df.y)
    xy_map = np.zeros((y_width+4, x_width+4), dtype=int)
    xy_map[df.y, df.x] = 1

    #---- Plotting 
    fig = plt.figure(1, figsize=(5, 5))
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
    plt.rc('font', family='Serif')

    cmap = sns.cubehelix_palette(start=1.2, hue=1, \
                                 light=1, rot=-1.05, as_cmap=True)

    ax = plt.subplot(1, 1, 1)
    xi = np.arange(x_width+4)
    yi = np.arange(y_width+4)

    x_com, y_com = calc_radius.calculate_com(df)
    r_d = calc_radius.calculate_radial_distance(df)
    r_g = calc_radius.calculate_geometric_r(df)

    C1 = sns.crayons['Mahogany']
    C2 = sns.crayons['Denim']
    c1 = plt.Circle((x_com, y_com), r_d, color=C1, fill=False, lw=1.25)
    c2 = plt.Circle((x_com, y_com), r_g, color=C2, fill=False, lw=2)
    ax.add_artist(c1)
    ax.add_artist(c2)

    plt.plot([x_com, x_com+r_d*np.sqrt(2)/2], \
             [y_com, y_com+r_d/np.sqrt(2)], color=C1, \
             lw=1.25, label=rf"Ave. Rad. Distance $r_d$ $\sim$ {r_d*25:.0f} m")
    plt.plot([x_com, x_com-r_g*np.sqrt(2)/2], 
             [y_com, y_com+r_g/np.sqrt(2)], color=C2, \
             lw=2, label=rf"Geometric Radius $r_g$ $\sim$ {r_g*25:.0f} m")
    plt.legend(fontsize=12, loc=2)

    im = plt.pcolormesh(xi, yi, xy_map, cmap=cmap, lw=0.5)

    plt.xticks(xi[::5], xi[::5] * 25 / 1e3)
    plt.yticks(yi[::5], yi[::5] * 25 / 1e3)
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')

    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)