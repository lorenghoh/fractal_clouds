import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

def pick_cid(cid, ctype):
    path = '/nodessd/loh/repos/tracking_parq'
    df = pq.read_table(f'{path}/clouds_00000121.pq', nthreads=6).to_pandas()
    df = df[(df.cid == cid) & (df.type == ctype)]

    # Calculate z index from coordinates
    df['z'] = df.coord // (256 * 256)

    # k_sample = df.z.value_counts().index[0]
    # df = df[(df.z == k_sample)]

    # From there, take the xy indices
    xy = df.coord % (256 * 256)
    df['y'] = xy // 256 
    df['x'] = xy % 256

    # Drop duplicates 
    df_ = df.drop_duplicates(subset=['y', 'x'], keep='first')

    x = df_.x.values
    y = df_.y.values

    print(df_.head())

    x_axis, y_axis = 256, 256
    if (max(x) - min(x)) > x_axis // 2:
        # Shift x-coordinates
        x = x + x_off
        x[x >= x_axis] = x[x >= x_axis] - x_axis

    if (max(y) - min(y)) > y_axis // 2:
        # Shift y-coordinates
        y = y + y_off
        y[y >= y_axis] = y[y >= y_axis] - y_axis

    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    print(f"\nCorresponding sub-domain size: {y_width}x{x_width}")
    print( "Adjusted coordinates: " \
          f"({min(y)}, {max(y)}), ({min(x)}, {max(x)})")

    # Map the projection onto a new 2D array (2x size)
    xy_map = np.zeros((y_width+4, x_width+4), dtype=int)
    x_sub = x - min(x) + 1
    y_sub = y - min(y) + 1
    xy_map[y_sub, x_sub] = 1

    return xy_map, pd.DataFrame({'x':x, 'y':y})
