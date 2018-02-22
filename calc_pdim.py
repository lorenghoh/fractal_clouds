import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pick_cloud_projection import pick_cid 

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    # Read horizontal slice from a cloud core
    xy_map, x, y = pick_cid(4563, 0)

    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    print(f"\nCorresponding sub-domain size: {y_width}x{x_width}")
    print( "Adjusted coordinates: " \
          f"({min(y)}, {max(y)}), ({min(x)}, {max(x)})")

    # Map the projection onto a new 2D array (2x size)
    Z = np.zeros((y_width+4, x_width+4), dtype=int)
    x = x - min(x) + 1
    y = y - min(y) + 1
    Z[y, x] = 1
    Z[Z > 0] = 1

    # Calculate perimeter and area
    area = np.sum(Z[Z > 0])
    p = np.sum(Z[:, 1:] != Z[:, :-1]) + np.sum(Z[1:, :] != Z[:-1, :])

    print(p * 25, area * 25*25)
    print("Dp =", 2*np.log10(p*25)/np.log10(area*25*25))