import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

from pick_cloud_projection import pick_cid 

def calculate_com(df):
    pass

def calculate_geometric_r(df):
    return np.sqrt(df.shape[0])

def calculate_radial_r():
    pass

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    # Read horizontal slice from a cloud core
    df = pick_cid(4563, 0)

    # Geometric radius

    # Average radial distance