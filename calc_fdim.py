import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

from pick_cloud_projection import pick_cid 

def count_box(xy_map, k):
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    # Read horizontal slice from a cloud core
    xy_map, x, y = pick_cid(6886, 4)

    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    print(f"\nCorresponding sub-domain size: {y_width}x{x_width}")
    print( "Adjusted coordinates: " \
          f"({min(y)}, {max(y)}), ({min(x)}, {max(x)})")

    # Map the projection onto a new 2D array (2x size)
    xy_map_sub = np.zeros((y_width*2, x_width*2), dtype=int)
    x_sub = x - min(x) + x_width // 2
    y_sub = y - min(y) + y_width // 2
    xy_map_sub[y_sub, x_sub] = 1
