import os, glob, warnings
import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

from pick_cloud_projection import pick_cid
import load_config
import calc_radius

from joblib import Parallel, delayed

c, config = load_config.c, load_config.config

def count_box(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)

    # We count non-empty (0) and non-full boxes (k*k)
    return len(np.where((S > 0) & (S < k*k))[0])

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
    xy_map = np.zeros((y_width+4, x_width+4), dtype=int)
    xy_map[y - min(y), x - min(x)] = 1

    return xy_map

def calculate_fdim(df):
    # Build sub-domain based on the tracking data
    xy_map = build_subdomain(df)

    # Leave only the perimeter of the cloud 
    xy_temp = np.roll(xy_map, 1, axis=0) \
            + np.roll(xy_map, -1, axis=0) \
            + np.roll(xy_map, 1, axis=1) \
            + np.roll(xy_map, -1, axis=1)
    xy_map[xy_temp == 4] = 0

    # Build successive box sizes (from 2**n down to 2**1)
    p = min(xy_map.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)

    # Scaling factor based on L/R
    # r_g = calc_radius.calculate_radial_distance(df)
    # r_d = calc_radius.calculate_geometric_r(df)
    # sizes = np.arange(int(r_d), 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(count_box(xy_map, size))
    
    # sizes = sizes / r_d
    # Fit the successive log(sizes) with log (counts)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            c = np.polyfit(np.log(sizes), np.log(counts), 1)
            return c, sizes, counts
        except:
            return [0, 0], 0, 0

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
        df = grp.filter(lambda x: x.size > 8)

        # Project the 3D cloud onto surface 
        # df = df.drop_duplicates(subset=['y', 'x'], keep='first')

        def calc_fdim_to_df(df):
            c, _, _ = calculate_fdim(df[-1])
            c = -c[0]
            if (c <= 1) | (c >= 2):
                return
            return pd.DataFrame({'fdim': [c]})

        group = df.groupby(['cid', 'z'], as_index=False)
        with Parallel(n_jobs=16) as Pr:
            result = Pr(delayed(calc_fdim_to_df)
                        (grouped) for grouped in group) 
            df = pd.concat(result, ignore_index=True)
            df.to_parquet(f'../pq/fdim_dump_{t:03d}.pq')
            