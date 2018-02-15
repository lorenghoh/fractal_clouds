import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

def count_box():
    pass

def index_to_zyx(index):
    z = index // (ny * nx)
    xy = index % (ny * nx)
    y = xy // nx
    x = xy % nx
    return pd.DataFrame({'z':z, 'y':y, 'x':x})

if __name__ == '__main__':
    # Calculate fdim from a sample cloud 
    path = '/nodessd/loh/temp'
    df = pq.read_table(f'{path}/clouds.pq', nthreads=6).to_pandas()
    df = df[(df.cid == 11023) & (df.type == 4)]

    # Calculate zyx indices from coordinates
    xy = df.coord % (256 * 256)
    df['z'] = df.coord // (256 * 256)
    df['y'] = xy // 256 
    df['x'] = xy % 256

    print(df[(df.z > 30) & (df.z < 40)].sort_values(by=['z']))