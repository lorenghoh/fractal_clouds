import os, glob, warnings
import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq

from sklearn import linear_model as lm

from pick_cloud_projection import pick_cid
import load_config
import calc_radius

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

    # Filter out noise
    xy_temp = np.roll(xy_map, 1, axis=0) \
            + np.roll(xy_map, -1, axis=0) \
            + np.roll(xy_map, 1, axis=1) \
            + np.roll(xy_map, -1, axis=1)
    xy_map[xy_temp == 0] = 0

    return xy_map

def find_perimeter(xy_map):
    # Leave only the perimeter of the cloud 
    xy_temp = np.roll(xy_map, 1, axis=0) \
            + np.roll(xy_map, -1, axis=0) \
            + np.roll(xy_map, 1, axis=1) \
            + np.roll(xy_map, -1, axis=1)
    xy_map[xy_temp == 4] = 0

    return xy_map

# Given a field, calculate perimeter
def calc_perimeter(Z):
    return np.sum(Z[:, 1:] != Z[:, :-1]) + \
                  np.sum(Z[1:, :] != Z[:-1, :])

def calculate_fdim(df):
    # Build sub-domain based on the tracking data
    xy_map = build_subdomain(df)
    xy_map = find_perimeter(xy_map)

    def count_box(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                            np.arange(0, Z.shape[1], k), axis=1)

        # Count non-empty (0) and non-full (k*k) boxes
        return len(np.where((S > 0) & (S < k*k))[0])

    # Scaling factor based on L/R
    # r_ = calc_radius.calculate_radial_distance(df)
    r_ = calc_radius.calculate_geometric_r(df)
    sizes = np.arange(int(r_), 1, -1)
    if len(sizes) < 6:
        return 0

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(count_box(xy_map, size))
    
    sizes = sizes / r_
    # Fit the successive log10(sizes) with log10(counts)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            # Fit the successive log10(sizes) with log10 (counts)
            model = lm.BayesianRidge()
            X = np.log10(sizes)[:, None]
            model.fit(X, np.log10(counts))
            return -model.coef_[0]
        except:
            return 0

def calculate_pdim(df):
    # Build sub-domain based on the tracking data
    xy_map = build_subdomain(df)

    # Given a field, return coarse observation
    def observe_coarse_field(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                            np.arange(0, Z.shape[1], k), axis=1)

        # Normalize coarse observation
        S[S > 0] = 1
        # Rebuild sampling map for perim. calc.
        S_ = np.zeros((S.shape[0]+2, S.shape[1]+2))
        S_[1:-1, 1:-1] = S[:]
        return S_

    # Scaling factor based on L/R
    # r_ = calc_radius.calculate_radial_distance(df)
    r_ = calc_radius.calculate_geometric_r(df)
    sizes = np.arange(int(r_/2), 0, -1)
    if len(sizes) < 4:
        return 0, 0

    area = np.sum(xy_map) * c.dx**2
    p = np.sum(calc_perimeter(xy_map)) * c.dx
    a_ = np.log10(area) / np.log10(p)

    X_p, Y_p = [], []
    for size in sizes:
        # Coefficient for horizontal scale
        C = c.dx * size

        Z = observe_coarse_field(xy_map, size)
        if np.sum(Z) == 0:
            continue
        p = np.sum(find_perimeter(Z)) * C

        X_p.append(C)
        Y_p.append(p)

    if len(np.unique(Y_p)) < 4:
        return a_, 0
    X_p = X_p / r_
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            # Fit the successive log10(X_p) with log10(Y_p)
            model = lm.BayesianRidge()
            X = np.log10(X_p)[:, None]
            model.fit(X, np.log10(Y_p))
            return a_, -model.coef_[0]
        except:
            return 0, 0

if __name__ == '__main__':
    filelist = sorted(glob.glob(f"{config['tracking']}/clouds_*.pq"))

    # Assert dataset integrity
    assert len(filelist) == c.nt

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
    
        grp = df.groupby(['cid', 'z'], as_index=False)
        df = grp.filter(lambda x: x.size > 0)

        def calc_fractality(df):
            f_d = calculate_fdim(df)
            a_d, p_d = calculate_pdim(df)
            return pd.DataFrame({'fdim': [f_d],
                                 'pdim': [p_d],
                                 'adim': [a_d]})

        group = df.groupby(['cid', 'z'], as_index=False)
        with Parallel(n_jobs=16) as Pr:
            result = Pr(delayed(calc_fractality)
                        (grouped) for _, grouped in group) 
            df = pd.concat(result, ignore_index=True)
            df.to_parquet(f'../pq/fdim_dump_{t:03d}.pq')
            