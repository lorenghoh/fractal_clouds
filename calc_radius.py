import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

from pick_cloud_projection import pick_cid 

# Return (x, y) tuple of COM point
def calculate_com(df):
    return df.x.mean(), df.y.mean()

def calculate_geometric_r(df):
    return np.sqrt(df.shape[0]/np.pi)

def calculate_radial_distance(df):
    x_com, y_com = calculate_com(df)

    del_x = df.x - x_com
    del_y = df.y - y_com
    
    return np.mean(np.sqrt(del_x**2 + del_y**2))

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    # Read horizontal slice from a cloud core
    df = pick_cid(4563, 0)

    # Print COM
    print(calculate_com(df))

    # Geometric radius
    r_g = calculate_geometric_r(df)

    # Average radial distance
    r_d = calculate_radial_distance(df)

    print(r_g)
    print(r_d)
