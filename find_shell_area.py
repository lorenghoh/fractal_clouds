import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pick_cloud_projection import pick_cid 
import calc_radius

# Given a field, return coarse observation
def observe_coarse_field(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)

    # Normalize coarse observation
    S[S > 0] = 1
    return S

if __name__ == '__main__':
    df = pick_cid(4563, 0)

    x_width = max(df.x) - min(df.x)
    y_width = max(df.y) - min(df.y)
    xy_map = np.zeros((y_width+4, x_width+4), dtype=int)
    xy_map[df.y, df.x] = 1

    # Radius estimates 
    r_g = calc_radius.calculate_radial_distance(df)
    r_d = calc_radius.calculate_geometric_r(df)
    sizes = np.arange(int(r_g), 1, -1)

    area_list = []
    for size in sizes:
        Z = observe_coarse_field(xy_map, size)
        area_list.append(np.sum(Z))
    print(area_list[-1] - area_list)
