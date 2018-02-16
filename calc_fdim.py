import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

from pick_cloud_projection import pick_cid 

def count_box():
    pass

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    # Read horizontal slice from a cloud core
    xy_map, x, y = pick_cid(6886, 4)

    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    print(f"\nCorresponding sub-domain size: {y_width}x{x_width}")
    print( "Adjusted coordinates: " \
          f"({min(y)}, {max(y)}), ({min(x)}, {max(x)})")

    # Map the projection onto a new 2D array
    xy_map_sub = np.zeros((x_width*2, y_width*2), dtype=int)
    x_sub = x - min(x) + x_width // 2
    y_sub = y - min(y) + y_width // 2
    xy_map_sub[y_sub, x_sub] = 1

