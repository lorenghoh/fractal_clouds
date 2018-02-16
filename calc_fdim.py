import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

def count_box():
    pass

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    path = '/nodessd/loh/temp'
    df = pq.read_table(f'{path}/clouds.pq', nthreads=6).to_pandas()
    df = df[(df.cid == 6886) & (df.type == 4)]

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

    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    print(f"\nCorresponding sub-domain size: {y_width}x{x_width}")
    print( "Adjusted coordinates: " \
          f"({min(y)}, {max(y)}), ({min(x)}, {max(x)})")

    # Then map it onto a new 2D array
    print(xy_map[min(y):max(y), min(x):max(x)])