import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

from pick_cloud_projection import pick_cid 

# Return (x, y) tuple of COM point
def calculate_com(x, y):
    return np.mean(x), np.mean(y)

# Let the functions take either the 2D Numpy array
# or a Pandas Dataframe; raise otherwise.
def get_xy_index(d_in):
    if isinstance(d_in, np.ndarray):
        return np.where(d_in == 1)
    elif isinstance(d_in, pd.DataFrame):
        return d_in.x, d_in.y
    else:
        raise("Unknown Input Type")

def calculate_geometric_r(d_in):
    x, y = get_xy_index(d_in)
    assert (len(x) == len(y))
    return np.sqrt(len(x)/np.pi)

def calculate_radial_distance(d_in):
    x, y = get_xy_index(d_in)
    x_c, y_c = calculate_com(x, y)
    
    return np.mean(np.sqrt((x - x_c)**2 + (y - y_c)**2))

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
