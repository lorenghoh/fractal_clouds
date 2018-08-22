import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq

from load_config import c, config

def pick_cid(cid, ctype):
    pq_file = f"{config['tracking']}/clouds_00000121.pq"
    df = pq.read_table(pq_file, nthreads=16).to_pandas()
    df = df[(df.cid == cid) & (df.type == ctype)]

    # Calculate z index from coordinates
    df['z'] = df.coord // (c.nx * c.ny)

    # From there, take the xy indices
    xy = df.coord % (c.nx * c.ny)
    df['y'] = xy // c.nx
    df['x'] = xy % c.nx

    # # Project the 3D cloud onto surface 
    # df_ = df.drop_duplicates(subset=['y', 'x'], keep='first')

    # Or, pick at the largest area
    df_ = df[(df.z == df.z.value_counts().index[0])]
    
    x = df_.x.values
    y = df_.y.values

    print(df_.head())

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
    print(f"\nCorresponding sub-domain size: {y_width}x{x_width}")
    print( "Adjusted coordinates: " \
          f"({min(y)}, {max(y)}), ({min(x)}, {max(x)})")

    x_sub = x - min(x) + 1
    y_sub = y - min(y) + 1

    return pd.DataFrame({'x':x_sub, 'y':y_sub})
