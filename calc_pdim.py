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

# Given a field, calculate perimeter
def calc_perimeter(Z):
    return np.sum(Z[:, 1:] != Z[:, :-1]) + \
           np.sum(Z[1:, :] != Z[:-1, :])

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    # Read horizontal slice from a cloud core
    df = pick_cid(4563, 0)

    x_width = max(df.x) - min(df.x)
    y_width = max(df.y) - min(df.y)
    xy_map = np.zeros((y_width+4, x_width+4), dtype=int)
    xy_map[df.y, df.x] = 1

    # Scaling factor based on L/R
    r_g = calc_radius.calculate_radial_distance(df)
    r_d = calc_radius.calculate_geometric_r(df)
    sizes = np.arange(int(r_g), 0, -1)

    Dp = []
    for size in sizes:
        # Coefficient for horizontal scale
        C = 25 * size

        Z = observe_coarse_field(xy_map, size)
        area = np.sum(Z) * C**2
        p = calc_perimeter(Z) * C

        Dp.append(p)
    sizes = sizes 

    print(sizes, Dp)

    # Fit the successive log(sizes) with log (counts)
    c = np.polyfit(np.log(sizes), np.log(Dp), 1)
    print(-c[0])

    # Calculate perimeter and area
    area = np.sum(xy_map[xy_map > 0]) * 25**2
    p = calc_perimeter(xy_map) * 25

    print(p, area)
    print("Dp =", 2*np.log10(p)/np.log10(area))