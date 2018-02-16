import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

def pick_cid(cid, ctype):
    path = '/nodessd/loh/temp'
    df = pq.read_table(f'{path}/clouds.pq', nthreads=6).to_pandas()
    df = df[(df.cid == cid) & (df.type == ctype)]

    # Calculate z index from coordinates
    df['z'] = df.coord // (256 * 256)

    k_sample = df.z.value_counts().index[0]
    df = df[(df.z == k_sample)]

    # From there, take the xy indices
    xy = df.coord % (256 * 256)
    df['y'] = xy // 256 
    df['x'] = xy % 256

    x = df.x.values
    y = df.y.values

    print(df.head())

    # Map cloud core onto the BOMEX domain and adjust
    xy_map = np.zeros((256, 256), dtype=int)
    xy_map[y, x] = 1

    x_axis, y_axis = xy_map.shape
    if (max(x) - min(x)) > x_axis // 2:
        # Shift target array
        x_off = x_axis - min(x[(x > x_axis // 2)])
        xy_map = np.roll(xy_map, x_off, axis=1)

        # Shift x-coordinates
        x = x + x_off
        x[x >= x_axis] = x[x >= x_axis] - x_axis

    if (max(y) - min(y)) > y_axis // 2:
        # Shift target array
        y_off = y_axis - min(y[(y > y_axis // 2)])
        xy_map = np.roll(xy_map, y_off, axis=0)
        
        # Shift y-coordinates
        y = y + y_off
        y[y >= y_axis] = y[y >= y_axis] - y_axis

    return xy_map, x, y

