import os, glob, warnings
import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq

from sklearn import linear_model as lm

from pick_cloud_projection import pick_cid
import load_config
import calc_radius

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed

c, config = load_config.c, load_config.config

def build_subdomain(df):
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

    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    xy_map = np.zeros((y_width+1, x_width+1), dtype=int)
    xy_map[y - min(y), x - min(x)] = 1

    S_ = np.zeros((xy_map.shape[0]+4, xy_map.shape[1]+4))
    S_[2:-2, 2:-2] = xy_map[:]
    return S_

# Given a field, calculate perimeter
def calc_perimeter(Z):
    return np.sum(Z[:, 1:] != Z[:, :-1]) + \
        np.sum(Z[1:, :] != Z[:-1, :])

def calculate_parameters(df):
    # Build sub-domain based on the tracking data
    xy_map = build_subdomain(df)
    area = np.sum(xy_map) * c.dx**2
    perimeter = calc_perimeter(xy_map) * c.dx
    return area, perimeter

def dump_dataset():
    filelist = sorted(glob.glob(f"{config['tracking']}/clouds_*.pq"))

    for t, f in enumerate(filelist):
        print(f'\t {t}/{len(filelist)} ({t/len(filelist)*100:.1f} %)', end='\r')

        # Read to Pandas Dataframe and process
        df = pq.read_pandas(f, nthreads=16).to_pandas()

        # Translate indices to coordinates
        df['z'] = df.coord // (c.nx * c.ny)
        xy = df.coord % (c.nx * c.ny)
        df['y'] = xy // c.ny 
        df['x'] = xy % c.nx

        # Take cloud regions and trim noise
        df = df[df.type == 0]

        def calc_fractality(df):
            a_, p_ = calculate_parameters(df)
            return pd.DataFrame({'a': [a_],
                                 'p': [p_]})

        group = df.groupby(['cid'], as_index=False)
        with Parallel(n_jobs=16) as Pr:
            result = Pr(delayed(calc_fractality)
                        (grouped) for _, grouped in group) 
            df = pd.concat(result, ignore_index=True)
            df.to_parquet(f'../pq/fdim_hres_ap_dump_{t:03d}.pq')

if __name__ == '__main__':
    dump_dataset()

    filelist = sorted(glob.glob(f"../pq/fdim_hres_ap_dump_*.pq"))
    df = pq.ParquetDataset(filelist).read(nthreads=16).to_pandas()

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

    x = df.p
    y = df.a
    m_ = (df.a > 0) & (df.p > 0)

    model = lm.BayesianRidge()
    X = np.log10(x[m_])[:, None]
    model.fit(X, np.log10(y[m_]))
    print(model.coef_[0])

    xmin, xmax = np.min(np.log10(x[m_])), np.max(np.log10(x[m_]))
    xi = np.linspace(xmin-0.1, xmax+0.1, 50)
    y_fit = model.predict(xi[:, None])

    plt.plot(xi, y_fit, 'r--')
    plt.plot(xi, 1.5*xi-0.5, 'k--')

    plt.scatter(np.log10(x[m_]), np.log10(y[m_]))

    plt.xlabel(r'$\log_{10}$ Perimeter [m]')
    plt.ylabel(r'$\log_{10}$ Area [m$^2$]')

    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)